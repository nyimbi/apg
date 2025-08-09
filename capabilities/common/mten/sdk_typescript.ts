/**
 * MTen TypeScript SDK
 * 
 * Company: Datacraft
 * Copyright: © 2025
 * Author: Nyimbi Odero
 * 
 * Comprehensive TypeScript SDK for Multi-Tenant Management (MTen) capability
 * with full type safety, modern async/await patterns, and browser/Node.js support.
 */

// Types and Enums

export enum TenantStatus {
  ACTIVE = 'active',
  SUSPENDED = 'suspended',
  PENDING = 'pending',
  ARCHIVED = 'archived'
}

export enum TenantTier {
  FREE = 'free',
  STANDARD = 'standard',
  PREMIUM = 'premium',
  ENTERPRISE = 'enterprise'
}

export enum DeploymentStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  ROLLED_BACK = 'rolled_back'
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  requestId: string;
  timestamp: string;
}

export interface Tenant {
  id: string;
  name: string;
  displayName: string;
  status: TenantStatus;
  tier: TenantTier;
  createdAt: string;
  updatedAt: string;
  configuration?: Record<string, any>;
  metadata?: Record<string, any>;
  resourceUsage?: Record<string, any>;
}

export interface TenantTemplate {
  id: string;
  name: string;
  displayName: string;
  description: string;
  category: string;
  version: string;
  configuration: Record<string, any>;
  resourceRequirements: Record<string, any>;
  createdAt: string;
  isPublic: boolean;
  tags: string[];
}

export interface CreateTenantRequest {
  name: string;
  tier: TenantTier;
  displayName?: string;
  templateId?: string;
  configuration?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface UpdateTenantRequest {
  displayName?: string;
  tier?: TenantTier;
  configuration?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface DeploymentResult {
  id: string;
  tenantId: string;
  status: DeploymentStatus;
  strategy: string;
  version: string;
  startedAt: string;
  completedAt?: string;
  logs: string[];
  rollbackAvailable: boolean;
}

export interface AnalyticsMetrics {
  tenantId: string;
  timestamp: string;
  cpuUsagePercent: number;
  memoryUsageMb: number;
  storageUsageGb: number;
  requestCount: number;
  errorRate: number;
  responseTimeMs: number;
  activeUsers: number;
}

export interface ListTenantsOptions {
  status?: TenantStatus;
  tier?: TenantTier;
  limit?: number;
  offset?: number;
}

export interface ListTemplatesOptions {
  category?: string;
  publicOnly?: boolean;
  limit?: number;
}

export interface GetMetricsOptions {
  startTime?: string;
  endTime?: string;
  interval?: string;
}

export interface MTenClientOptions {
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  userAgent?: string;
}

// Custom Error Classes

export class MTenSDKError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public responseData?: Record<string, any>
  ) {
    super(message);
    this.name = 'MTenSDKError';
  }
}

export class AuthenticationError extends MTenSDKError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'AuthenticationError';
  }
}

export class ValidationError extends MTenSDKError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'ValidationError';
  }
}

export class NetworkError extends MTenSDKError {
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

// HTTP Client abstraction for cross-platform support

interface HTTPClient {
  request<T = any>(
    method: string,
    url: string,
    options?: {
      headers?: Record<string, string>;
      body?: string;
      timeout?: number;
    }
  ): Promise<{
    status: number;
    data: T;
    headers: Record<string, string>;
  }>;
}

class FetchHTTPClient implements HTTPClient {
  async request<T = any>(
    method: string,
    url: string,
    options: {
      headers?: Record<string, string>;
      body?: string;
      timeout?: number;
    } = {}
  ): Promise<{
    status: number;
    data: T;
    headers: Record<string, string>;
  }> {
    const controller = new AbortController();
    const timeoutId = options.timeout 
      ? setTimeout(() => controller.abort(), options.timeout)
      : null;

    try {
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        body: options.body,
        signal: controller.signal,
      });

      const data = await response.json();
      const headers: Record<string, string> = {};
      
      response.headers.forEach((value, key) => {
        headers[key] = value;
      });

      return {
        status: response.status,
        data,
        headers,
      };
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new NetworkError('Request timeout');
      }
      throw new NetworkError(`Network request failed: ${error.message}`);
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    }
  }
}

// Utility Functions

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Main SDK Client

export class MTenClient {
  private baseUrl: string;
  private apiKey: string;
  private httpClient: HTTPClient;
  private options: Required<MTenClientOptions>;

  constructor(
    baseUrl: string,
    apiKey: string,
    options: MTenClientOptions = {}
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.httpClient = new FetchHTTPClient();
    this.options = {
      timeout: options.timeout ?? 30000,
      retryAttempts: options.retryAttempts ?? 3,
      retryDelay: options.retryDelay ?? 1000,
      userAgent: options.userAgent ?? `MTen-TypeScript-SDK/1.0.0`,
    };
  }

  private async request<T = any>(
    method: string,
    endpoint: string,
    data?: Record<string, any>,
    params?: Record<string, any>
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}/api/v1${endpoint}`);
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
      'User-Agent': this.options.userAgent,
      'X-Request-ID': generateRequestId(),
    };

    for (let attempt = 0; attempt <= this.options.retryAttempts; attempt++) {
      try {
        const response = await this.httpClient.request<T>(method, url.toString(), {
          headers,
          body: data ? JSON.stringify(data) : undefined,
          timeout: this.options.timeout,
        });

        if (response.status >= 400) {
          const errorData = response.data as any;
          const message = errorData?.message || `HTTP ${response.status}`;

          if (response.status === 401) {
            throw new AuthenticationError(
              'Invalid API key or authentication failed',
              response.status,
              errorData
            );
          } else if (response.status === 422) {
            throw new ValidationError(message, response.status, errorData);
          } else {
            throw new MTenSDKError(message, response.status, errorData);
          }
        }

        return response.data;
      } catch (error) {
        if (error instanceof MTenSDKError || attempt === this.options.retryAttempts) {
          throw error;
        }

        await sleep(this.options.retryDelay * Math.pow(2, attempt));
      }
    }

    throw new MTenSDKError('Unexpected error in request processing');
  }

  // Tenant Management Methods

  /**
   * List tenants with optional filtering
   */
  async listTenants(options: ListTenantsOptions = {}): Promise<APIResponse<Tenant[]>> {
    const params: Record<string, any> = {
      limit: options.limit ?? 100,
      offset: options.offset ?? 0,
    };

    if (options.status) params.status = options.status;
    if (options.tier) params.tier = options.tier;

    const response = await this.request<APIResponse<Tenant[]>>(
      'GET',
      '/tenants',
      undefined,
      params
    );

    return response;
  }

  /**
   * Get tenant by ID
   */
  async getTenant(tenantId: string): Promise<APIResponse<Tenant>> {
    const response = await this.request<APIResponse<Tenant>>(
      'GET',
      `/tenants/${tenantId}`
    );

    return response;
  }

  /**
   * Create new tenant
   */
  async createTenant(request: CreateTenantRequest): Promise<APIResponse<Tenant>> {
    const data = {
      name: request.name,
      tier: request.tier,
      displayName: request.displayName ?? request.name,
      templateId: request.templateId,
      configuration: request.configuration ?? {},
      metadata: request.metadata ?? {},
    };

    const response = await this.request<APIResponse<Tenant>>(
      'POST',
      '/tenants',
      data
    );

    return response;
  }

  /**
   * Update existing tenant
   */
  async updateTenant(
    tenantId: string,
    request: UpdateTenantRequest
  ): Promise<APIResponse<Tenant>> {
    const response = await this.request<APIResponse<Tenant>>(
      'PATCH',
      `/tenants/${tenantId}`,
      request
    );

    return response;
  }

  /**
   * Delete tenant
   */
  async deleteTenant(tenantId: string, force = false): Promise<APIResponse<boolean>> {
    const params = force ? { force: 'true' } : {};

    const response = await this.request<APIResponse<boolean>>(
      'DELETE',
      `/tenants/${tenantId}`,
      undefined,
      params
    );

    return response;
  }

  // Template Management Methods

  /**
   * List available tenant templates
   */
  async listTemplates(options: ListTemplatesOptions = {}): Promise<APIResponse<TenantTemplate[]>> {
    const params: Record<string, any> = {
      limit: options.limit ?? 50,
      publicOnly: options.publicOnly ?? true,
    };

    if (options.category) params.category = options.category;

    const response = await this.request<APIResponse<TenantTemplate[]>>(
      'GET',
      '/templates',
      undefined,
      params
    );

    return response;
  }

  /**
   * Get template by ID
   */
  async getTemplate(templateId: string): Promise<APIResponse<TenantTemplate>> {
    const response = await this.request<APIResponse<TenantTemplate>>(
      'GET',
      `/templates/${templateId}`
    );

    return response;
  }

  /**
   * Create new tenant template
   */
  async createTemplate(template: Omit<TenantTemplate, 'id' | 'createdAt'>): Promise<APIResponse<TenantTemplate>> {
    const response = await this.request<APIResponse<TenantTemplate>>(
      'POST',
      '/templates',
      template
    );

    return response;
  }

  // Deployment Methods

  /**
   * Deploy tenant with specified strategy
   */
  async deployTenant(
    tenantId: string,
    version?: string,
    strategy = 'rolling'
  ): Promise<APIResponse<DeploymentResult>> {
    const data = {
      tenantId,
      strategy,
      ...(version && { version }),
    };

    const response = await this.request<APIResponse<DeploymentResult>>(
      'POST',
      '/deployments',
      data
    );

    return response;
  }

  /**
   * Get deployment status
   */
  async getDeploymentStatus(deploymentId: string): Promise<APIResponse<DeploymentResult>> {
    const response = await this.request<APIResponse<DeploymentResult>>(
      'GET',
      `/deployments/${deploymentId}`
    );

    return response;
  }

  /**
   * Rollback deployment to previous version
   */
  async rollbackDeployment(
    deploymentId: string,
    targetVersion?: string
  ): Promise<APIResponse<DeploymentResult>> {
    const data = targetVersion ? { targetVersion } : {};

    const response = await this.request<APIResponse<DeploymentResult>>(
      'POST',
      `/deployments/${deploymentId}/rollback`,
      data
    );

    return response;
  }

  // Analytics Methods

  /**
   * Get tenant analytics metrics
   */
  async getTenantMetrics(
    tenantId: string,
    options: GetMetricsOptions = {}
  ): Promise<APIResponse<AnalyticsMetrics[]>> {
    const params: Record<string, any> = {
      interval: options.interval ?? '1h',
    };

    if (options.startTime) params.startTime = options.startTime;
    if (options.endTime) params.endTime = options.endTime;

    const response = await this.request<APIResponse<AnalyticsMetrics[]>>(
      'GET',
      `/tenants/${tenantId}/metrics`,
      undefined,
      params
    );

    return response;
  }

  /**
   * Get tenant health score
   */
  async getTenantHealthScore(tenantId: string): Promise<APIResponse<number>> {
    const response = await this.request<APIResponse<{ healthScore: number }>>(
      'GET',
      `/tenants/${tenantId}/health`
    );

    return {
      ...response,
      data: response.data?.healthScore ?? 0,
    };
  }

  // Real-time Methods

  /**
   * Stream real-time tenant events (Server-Sent Events)
   */
  streamTenantEvents(tenantIds?: string[]): EventSource {
    const url = new URL(`${this.baseUrl}/api/v1/tenants/stream`);
    
    if (tenantIds?.length) {
      url.searchParams.append('tenantIds', tenantIds.join(','));
    }

    // Add authorization header for EventSource (browser limitation workaround)
    const eventSource = new EventSource(url.toString());

    return eventSource;
  }

  /**
   * Stream deployment logs using WebSocket
   */
  streamDeploymentLogs(deploymentId: string): WebSocket {
    const wsUrl = this.baseUrl.replace(/^https?/, 'wss');
    const url = `${wsUrl}/api/v1/deployments/${deploymentId}/logs/stream?auth=${encodeURIComponent(this.apiKey)}`;

    const ws = new WebSocket(url);

    return ws;
  }

  // Utility Methods

  /**
   * Ping the API to check connectivity
   */
  async ping(): Promise<APIResponse<Record<string, any>>> {
    const response = await this.request<APIResponse<Record<string, any>>>(
      'GET',
      '/ping'
    );

    return response;
  }

  /**
   * Get API version and capability information
   */
  async getApiInfo(): Promise<APIResponse<Record<string, any>>> {
    const response = await this.request<APIResponse<Record<string, any>>>(
      'GET',
      '/info'
    );

    return response;
  }
}

// High-level convenience functions

/**
 * Create and configure MTen client
 */
export function createMTenClient(
  baseUrl: string,
  apiKey: string,
  options?: MTenClientOptions
): MTenClient {
  return new MTenClient(baseUrl, apiKey, options);
}

/**
 * Quick tenant setup with sensible defaults
 */
export async function quickTenantSetup(
  client: MTenClient,
  name: string,
  tier: TenantTier,
  templateName?: string
): Promise<Tenant> {
  let templateId: string | undefined;

  if (templateName) {
    const templatesResponse = await client.listTemplates();
    if (templatesResponse.success && templatesResponse.data) {
      const template = templatesResponse.data.find(t => t.name === templateName);
      if (template) {
        templateId = template.id;
      }
    }
  }

  const tenantResponse = await client.createTenant({
    name,
    tier,
    templateId,
  });

  if (!tenantResponse.success || !tenantResponse.data) {
    throw new MTenSDKError(
      `Failed to create tenant: ${tenantResponse.error}`
    );
  }

  return tenantResponse.data;
}

// React Hook for easy integration (optional, requires React)
declare const React: any;

export interface UseMTenOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

/**
 * React hook for MTen integration (requires React)
 */
export function useMTen(
  client: MTenClient,
  options: UseMTenOptions = {}
) {
  if (typeof React === 'undefined') {
    throw new Error('React is required to use useMTen hook');
  }

  const [tenants, setTenants] = React.useState<Tenant[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const refreshTenants = React.useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await client.listTenants();
      if (response.success && response.data) {
        setTenants(response.data);
      } else {
        setError(response.error || 'Failed to load tenants');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [client]);

  React.useEffect(() => {
    refreshTenants();

    if (options.autoRefresh) {
      const interval = setInterval(
        refreshTenants,
        options.refreshInterval || 30000
      );
      return () => clearInterval(interval);
    }
  }, [refreshTenants, options.autoRefresh, options.refreshInterval]);

  return {
    tenants,
    loading,
    error,
    refreshTenants,
  };
}

// Export version
export const version = '1.0.0';