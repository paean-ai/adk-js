/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Blob,
  createPartFromText,
  FileData,
  FinishReason,
  GenerateContentResponse,
  GoogleGenAI,
  Part,
} from '@google/genai';

import {logger} from '../utils/logger.js';
import {isGemini3PreviewModel} from '../utils/model_name.js';
import {GoogleLLMVariant} from '../utils/variant_utils.js';

import {BaseLlm} from './base_llm.js';
import {BaseLlmConnection} from './base_llm_connection.js';
import {GeminiLlmConnection} from './gemini_llm_connection.js';
import {LlmRequest} from './llm_request.js';
import {createLlmResponse, LlmResponse} from './llm_response.js';

const AGENT_ENGINE_TELEMETRY_TAG = 'remote_reasoning_engine';
const AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_AGENT_ENGINE_ID';

/**
 * Default API endpoint for Gemini 3 preview models.
 * Gemini 3 preview models require the aiplatform.googleapis.com endpoint with
 * the publishers/google path prefix.
 *
 * The SDK constructs URLs like: ${baseUrl}/models/${model}:generateContent
 * But Gemini 3 preview needs: https://aiplatform.googleapis.com/v1/publishers/google/models/${model}:generateContent
 *
 * So we set the baseUrl to include the path prefix up to (but not including) /models/
 */
const GEMINI3_PREVIEW_API_ENDPOINT =
  'https://aiplatform.googleapis.com/v1/publishers/google';

/**
 * The parameters for creating a Gemini instance.
 */
export interface GeminiParams {
  /**
   * The name of the model to use. Defaults to 'gemini-2.5-flash'.
   */
  model?: string;
  /**
   * The API key to use for the Gemini API. If not provided, it will look for
   * the GOOGLE_GENAI_API_KEY or GEMINI_API_KEY environment variable.
   */
  apiKey?: string;
  /**
   * Whether to use Vertex AI. If true, `project`, `location`
   * should be provided.
   */
  vertexai?: boolean;
  /**
   * The Vertex AI project ID. Required if `vertexai` is true.
   */
  project?: string;
  /**
   * The Vertex AI location. Required if `vertexai` is true.
   */
  location?: string;
  /**
   * Headers to merge with internally crafted headers.
   */
  headers?: Record<string, string>;
  /**
   * Custom API endpoint URL. If not provided, uses the default endpoint
   * based on the model type:
   * - Gemini 3 preview models: aiplatform.googleapis.com
   * - Other models: uses SDK default (generativelanguage.googleapis.com)
   *
   * Can also be set via GEMINI_API_ENDPOINT environment variable.
   */
  apiEndpoint?: string;
}

/**
 * Integration for Gemini models.
 */
export class Gemini extends BaseLlm {
  private readonly apiKey?: string;
  private readonly vertexai: boolean;
  private readonly project?: string;
  private readonly location?: string;
  private readonly headers?: Record<string, string>;
  private readonly apiEndpoint?: string;
  private readonly isGemini3Preview: boolean;

  /**
   * @param params The parameters for creating a Gemini instance.
   */
  constructor({
    model,
    apiKey,
    vertexai,
    project,
    location,
    headers,
    apiEndpoint,
  }: GeminiParams) {
    if (!model) {
      model = 'gemini-2.5-flash';
    }

    super({model});

    this.project = project;
    this.location = location;
    this.apiKey = apiKey;
    this.headers = headers;
    this.isGemini3Preview = isGemini3PreviewModel(model);

    const canReadEnv = typeof process === 'object';

    // Handle API endpoint configuration
    this.apiEndpoint = apiEndpoint;
    if (!this.apiEndpoint && canReadEnv) {
      this.apiEndpoint = process.env['GEMINI_API_ENDPOINT'];
    }
    // For Gemini 3 preview models, use the aiplatform.googleapis.com endpoint by default
    if (!this.apiEndpoint && this.isGemini3Preview) {
      this.apiEndpoint = GEMINI3_PREVIEW_API_ENDPOINT;
      logger.info(`Using Gemini 3 preview endpoint: ${this.apiEndpoint}`);
    }

    // Determine vertexai mode from constructor or environment
    let useVertexAI = !!vertexai;
    if (!useVertexAI && canReadEnv) {
      const vertexAIfromEnv = process.env['GOOGLE_GENAI_USE_VERTEXAI'];
      if (vertexAIfromEnv) {
        useVertexAI =
          vertexAIfromEnv.toLowerCase() === 'true' || vertexAIfromEnv === '1';
      }
    }

    // For Gemini 3 preview models, force API key mode instead of Vertex AI.
    // This is because Gemini 3 preview requires a special endpoint path
    // (publishers/google/models/) that is not compatible with standard Vertex AI
    // SDK configuration. The custom apiEndpoint is only applied in non-Vertex AI mode.
    if (this.isGemini3Preview && useVertexAI) {
      // Check if API key is available before switching modes
      const availableApiKey =
        apiKey ||
        (canReadEnv
          ? process.env['GOOGLE_GENAI_API_KEY'] || process.env['GEMINI_API_KEY']
          : undefined);
      if (availableApiKey) {
        logger.info(
          'Gemini 3 preview detected with Vertex AI mode. Switching to API key mode for correct endpoint handling.',
        );
        useVertexAI = false;
        this.apiKey = availableApiKey;
      } else {
        logger.warn(
          'Gemini 3 preview requires API key authentication for correct endpoint handling. ' +
            'Set GEMINI_API_KEY or GOOGLE_GENAI_API_KEY environment variable for best compatibility.',
        );
      }
    }

    this.vertexai = useVertexAI;

    if (this.vertexai) {
      if (canReadEnv && !this.project) {
        this.project = process.env['GOOGLE_CLOUD_PROJECT'];
      }
      if (canReadEnv && !this.location) {
        this.location = process.env['GOOGLE_CLOUD_LOCATION'];
      }
      if (!this.project) {
        throw new Error(
          'VertexAI project must be provided via constructor or GOOGLE_CLOUD_PROJECT environment variable.',
        );
      }
      if (!this.location) {
        throw new Error(
          'VertexAI location must be provided via constructor or GOOGLE_CLOUD_LOCATION environment variable.',
        );
      }
    } else {
      if (!this.apiKey && canReadEnv) {
        this.apiKey =
          process.env['GOOGLE_GENAI_API_KEY'] || process.env['GEMINI_API_KEY'];
      }
      if (!this.apiKey) {
        throw new Error(
          'API key must be provided via constructor or GOOGLE_GENAI_API_KEY or GEMINI_API_KEY environment variable.',
        );
      }
    }
  }

  /**
   * A list of model name patterns that are supported by this LLM.
   *
   * @returns A list of supported models.
   */
  static override readonly supportedModels: Array<string | RegExp> = [
    /gemini-.*/,
    // fine-tuned vertex endpoint pattern
    /projects\/.+\/locations\/.+\/endpoints\/.+/,
    // vertex gemini long name
    /projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+/,
  ];

  private _apiClient?: GoogleGenAI;
  private _apiBackend?: GoogleLLMVariant;
  private _trackingHeaders?: Record<string, string>;
  private _liveApiVersion?: string;
  private _liveApiClient?: GoogleGenAI;

  /**
   * Sends a request to the Gemini model.
   *
   * @param llmRequest LlmRequest, the request to send to the Gemini model.
   * @param stream bool = false, whether to do streaming call.
   * @yields LlmResponse: The model response.
   */
  override async *generateContentAsync(
    llmRequest: LlmRequest,
    stream = false,
  ): AsyncGenerator<LlmResponse, void> {
    this.preprocessRequest(llmRequest);
    this.maybeAppendUserContent(llmRequest);
    logger.info(
      `Sending out request, model: ${llmRequest.model}, backend: ${
        this.apiBackend
      }, stream: ${stream}`,
    );

    if (llmRequest.config?.httpOptions) {
      llmRequest.config.httpOptions.headers = {
        ...llmRequest.config.httpOptions.headers,
        ...this.trackingHeaders,
      };
    }

    if (stream) {
      const streamResult = await this.apiClient.models.generateContentStream({
        model: llmRequest.model ?? this.model,
        contents: llmRequest.contents,
        config: llmRequest.config,
      });
      let thoughtText = '';
      let thoughtSignature: string | undefined;
      let text = '';
      let usageMetadata;
      let lastResponse: GenerateContentResponse | undefined;

      // TODO - b/425992518: verify the type of streaming response is correct.
      for await (const response of streamResult) {
        lastResponse = response;
        const llmResponse = createLlmResponse(response);
        usageMetadata = llmResponse.usageMetadata;
        const firstPart = llmResponse.content?.parts?.[0];
        // Accumulates the text and thought text from the first part.
        if (firstPart?.text) {
          if ('thought' in firstPart && firstPart.thought) {
            thoughtText += firstPart.text;
            // Preserve thoughtSignature from the thought parts for Gemini 3 thinking mode
            if ('thoughtSignature' in firstPart && firstPart.thoughtSignature) {
              thoughtSignature = firstPart.thoughtSignature as string;
            }
          } else {
            text += firstPart.text;
          }
          llmResponse.partial = true;
        } else if (
          (thoughtText || text) &&
          (!firstPart || !firstPart.inlineData)
        ) {
          // Flushes the data if there's no more text.
          const parts: Part[] = [];
          if (thoughtText) {
            // Include thoughtSignature when flushing accumulated thought text
            // This is required for Gemini 3 thinking mode to maintain signature chain
            const thoughtPart: Part = {text: thoughtText, thought: true};
            if (thoughtSignature) {
              thoughtPart.thoughtSignature = thoughtSignature;
            }
            parts.push(thoughtPart);
          }
          if (text) {
            parts.push(createPartFromText(text));
          }
          yield {
            content: {
              role: 'model',
              parts,
            },
            usageMetadata: llmResponse.usageMetadata,
          };
          thoughtText = '';
          thoughtSignature = undefined;
          text = '';
        }
        yield llmResponse;
      }
      if (
        (text || thoughtText) &&
        lastResponse?.candidates?.[0]?.finishReason === FinishReason.STOP
      ) {
        const parts: Part[] = [];
        if (thoughtText) {
          // Include thoughtSignature in final flush as well
          const thoughtPart: Part = {text: thoughtText, thought: true};
          if (thoughtSignature) {
            thoughtPart.thoughtSignature = thoughtSignature;
          }
          parts.push(thoughtPart);
        }
        if (text) {
          parts.push({text: text});
        }
        yield {
          content: {
            role: 'model',
            parts,
          },
          usageMetadata,
        };
      }
    } else {
      const response = await this.apiClient.models.generateContent({
        model: llmRequest.model ?? this.model,
        contents: llmRequest.contents,
        config: llmRequest.config,
      });
      yield createLlmResponse(response);
    }
  }

  get apiClient(): GoogleGenAI {
    if (this._apiClient) {
      return this._apiClient;
    }

    const combinedHeaders = {
      ...this.trackingHeaders,
      ...this.headers,
    };

    if (this.vertexai) {
      this._apiClient = new GoogleGenAI({
        vertexai: this.vertexai,
        project: this.project,
        location: this.location,
        httpOptions: {headers: combinedHeaders},
      });
    } else {
      // Build httpOptions with optional baseUrl for Gemini 3 preview models
      const httpOptions: Record<string, unknown> = {headers: combinedHeaders};
      if (this.apiEndpoint) {
        httpOptions.baseUrl = this.apiEndpoint;
        logger.debug(`Using custom API endpoint: ${this.apiEndpoint}`);

        // For Gemini 3 preview models on aiplatform.googleapis.com, we need to:
        // 1. Use the baseUrl that includes /v1/publishers/google path
        // 2. Prevent the SDK from adding its own API version prefix
        // The baseUrl already contains the version, so we don't need apiVersion
        if (this.isGemini3Preview) {
          // Set empty apiVersion to prevent SDK from adding version prefix
          // since the version is already included in the baseUrl
          httpOptions.apiVersion = '';
          logger.info(
            `Gemini 3 preview mode: using direct API path without version prefix`,
          );
        }
      }

      // Explicitly set vertexai: false to prevent SDK from auto-detecting
      // Vertex AI mode based on environment (e.g., GCP credentials on Cloud Run).
      // This ensures the API key authentication and custom baseUrl are used.
      this._apiClient = new GoogleGenAI({
        apiKey: this.apiKey,
        vertexai: false,
        httpOptions,
      });
    }
    return this._apiClient;
  }

  get apiBackend(): GoogleLLMVariant {
    if (!this._apiBackend) {
      this._apiBackend = this.apiClient.vertexai
        ? GoogleLLMVariant.VERTEX_AI
        : GoogleLLMVariant.GEMINI_API;
    }
    return this._apiBackend;
  }

  get liveApiVersion(): string {
    if (!this._liveApiVersion) {
      this._liveApiVersion =
        this.apiBackend === GoogleLLMVariant.VERTEX_AI ? 'v1beta1' : 'v1alpha';
    }
    return this._liveApiVersion;
  }

  get liveApiClient(): GoogleGenAI {
    if (!this._liveApiClient) {
      const httpOptions: Record<string, unknown> = {
        headers: this.trackingHeaders,
        apiVersion: this.liveApiVersion,
      };
      if (this.apiEndpoint) {
        httpOptions.baseUrl = this.apiEndpoint;

        // For Gemini 3 preview models, the baseUrl already contains the version
        // so we don't need the SDK to add another version prefix
        if (this.isGemini3Preview) {
          httpOptions.apiVersion = '';
        }
      }

      this._liveApiClient = new GoogleGenAI({
        apiKey: this.apiKey,
        httpOptions,
      });
    }
    return this._liveApiClient;
  }

  /**
   * Connects to the Gemini model and returns an llm connection.
   *
   * @param llmRequest LlmRequest, the request to send to the Gemini model.
   * @returns BaseLlmConnection, the connection to the Gemini model.
   */
  override async connect(llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    // add tracking headers to custom headers and set api_version given
    // the customized http options will override the one set in the api client
    // constructor
    if (llmRequest.liveConnectConfig?.httpOptions) {
      if (!llmRequest.liveConnectConfig.httpOptions.headers) {
        llmRequest.liveConnectConfig.httpOptions.headers = {};
      }
      Object.assign(
        llmRequest.liveConnectConfig.httpOptions.headers,
        this.trackingHeaders,
      );
      // For Gemini 3 preview, the baseUrl already contains the version
      // so we use empty apiVersion to prevent double version prefix
      llmRequest.liveConnectConfig.httpOptions.apiVersion = this
        .isGemini3Preview
        ? ''
        : this.liveApiVersion;
    }

    if (llmRequest.config?.systemInstruction) {
      llmRequest.liveConnectConfig.systemInstruction = {
        role: 'system',
        // TODO - b/425992518: validate type casting works well.
        parts: [
          createPartFromText(llmRequest.config.systemInstruction as string),
        ],
      };
    }

    llmRequest.liveConnectConfig.tools = llmRequest.config?.tools;

    const liveSession = await this.liveApiClient.live.connect({
      model: llmRequest.model ?? this.model,
      config: llmRequest.liveConnectConfig,
      callbacks: {
        // TODO - b/425992518: GenAI SDK inconsistent API, missing methods.
        onmessage: () => {},
      },
    });
    return new GeminiLlmConnection(liveSession);
  }

  private preprocessRequest(llmRequest: LlmRequest): void {
    if (this.apiBackend === GoogleLLMVariant.GEMINI_API) {
      if (llmRequest.config) {
        // Using API key from Google AI Studio to call model doesn't support
        // labels.
        (llmRequest.config as any).labels = undefined;
      }
      if (llmRequest.contents) {
        for (const content of llmRequest.contents) {
          if (!content.parts) continue;
          for (const part of content.parts) {
            removeDisplayNameIfPresent(part.inlineData);
            removeDisplayNameIfPresent(part.fileData);
          }
        }
      }
    }
  }
}

function removeDisplayNameIfPresent(
  dataObj: Blob | FileData | undefined,
): void {
  // display_name is not supported for Gemini API (non-vertex)
  if (dataObj && (dataObj as FileData).displayName) {
    (dataObj as FileData).displayName = undefined;
  }
}
