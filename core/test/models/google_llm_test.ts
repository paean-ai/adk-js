/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {Gemini, GeminiParams} from '@paean-ai/adk';

import {
  isGemini3Model,
  isGemini3PreviewModel,
} from '../../src/utils/model_name.js';
import {version} from '../../src/version.js';

class TestGemini extends Gemini {
  constructor(params: GeminiParams) {
    super(params);
  }
  getTrackingHeaders(): Record<string, string> {
    return this.trackingHeaders;
  }
}

describe('GoogleLlm', () => {
  it('should set tracking headers correctly when GOOGLE_CLOUD_AGENT_ENGINE_ID is not set',
     () => {
       const llm = new TestGemini({apiKey: 'test-key'});
       const headers = llm.getTrackingHeaders();
       const expectedValue =
           `google-adk/${version} gl-typescript/${process.version}`;
       expect(headers['x-goog-api-client']).toEqual(expectedValue);
       expect(headers['user-agent']).toEqual(expectedValue);
     });

  it('should set tracking headers correctly when GOOGLE_CLOUD_AGENT_ENGINE_ID is set',
     () => {
       process.env['GOOGLE_CLOUD_AGENT_ENGINE_ID'] = 'test-engine';
       const llm = new TestGemini({apiKey: 'test-key'});
       const headers = llm.getTrackingHeaders();
       const expectedValue = `google-adk/${
           version}+remote_reasoning_engine gl-typescript/${process.version}`;
       expect(headers['x-goog-api-client']).toEqual(expectedValue);
       expect(headers['user-agent']).toEqual(expectedValue);
     });
});

describe('Gemini 3 Model Detection', () => {
  it('should detect gemini-3-flash-preview as Gemini 3 model', () => {
    expect(isGemini3Model('gemini-3-flash-preview')).toBe(true);
    expect(isGemini3PreviewModel('gemini-3-flash-preview')).toBe(true);
  });

  it('should detect gemini-3-pro-preview as Gemini 3 model', () => {
    expect(isGemini3Model('gemini-3-pro-preview')).toBe(true);
    expect(isGemini3PreviewModel('gemini-3-pro-preview')).toBe(true);
  });

  it('should detect gemini-3-pro-image-preview as Gemini 3 preview model', () => {
    expect(isGemini3Model('gemini-3-pro-image-preview')).toBe(true);
    expect(isGemini3PreviewModel('gemini-3-pro-image-preview')).toBe(true);
  });

  it('should NOT detect gemini-2.5-flash as Gemini 3 model', () => {
    expect(isGemini3Model('gemini-2.5-flash')).toBe(false);
    expect(isGemini3PreviewModel('gemini-2.5-flash')).toBe(false);
  });

  it('should NOT detect gemini-1.5-pro as Gemini 3 model', () => {
    expect(isGemini3Model('gemini-1.5-pro')).toBe(false);
    expect(isGemini3PreviewModel('gemini-1.5-pro')).toBe(false);
  });

  it('should detect non-preview Gemini 3 models correctly', () => {
    // Hypothetical stable Gemini 3 model (when released)
    expect(isGemini3Model('gemini-3-flash')).toBe(true);
    expect(isGemini3PreviewModel('gemini-3-flash')).toBe(false);
  });
});

describe('Gemini 3 Instance Creation', () => {
  it('should create Gemini instance with gemini-3-flash-preview model', () => {
    const llm = new TestGemini({
      apiKey: 'test-key',
      model: 'gemini-3-flash-preview',
    });
    expect(llm).toBeDefined();
  });

  it('should create Gemini instance with custom apiEndpoint', () => {
    const llm = new TestGemini({
      apiKey: 'test-key',
      model: 'gemini-3-flash-preview',
      apiEndpoint: 'https://custom-endpoint.com',
    });
    expect(llm).toBeDefined();
  });
});
