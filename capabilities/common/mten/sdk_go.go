// Package mten provides a comprehensive Go SDK for Multi-Tenant Management (MTen) capability
// with high-performance, type-safe API client and extensive functionality.
//
// Company: Datacraft
// Copyright: © 2025
// Author: Nyimbi Odero
package mten

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// Version of the SDK
const Version = "1.0.0"

// Enums

// TenantStatus represents the status of a tenant
type TenantStatus string

const (
	TenantStatusActive    TenantStatus = "active"
	TenantStatusSuspended TenantStatus = "suspended"
	TenantStatusPending   TenantStatus = "pending"
	TenantStatusArchived  TenantStatus = "archived"
)

// TenantTier represents the tier level of a tenant
type TenantTier string

const (
	TenantTierFree       TenantTier = "free"
	TenantTierStandard   TenantTier = "standard"
	TenantTierPremium    TenantTier = "premium"
	TenantTierEnterprise TenantTier = "enterprise"
)

// DeploymentStatus represents the status of a deployment
type DeploymentStatus string

const (
	DeploymentStatusPending    DeploymentStatus = "pending"
	DeploymentStatusInProgress DeploymentStatus = "in_progress"
	DeploymentStatusCompleted  DeploymentStatus = "completed"
	DeploymentStatusFailed     DeploymentStatus = "failed"
	DeploymentStatusRolledBack DeploymentStatus = "rolled_back"
)

// Data Models

// APIResponse represents a generic API response
type APIResponse[T any] struct {
	Success   bool      `json:"success"`
	Data      *T        `json:"data,omitempty"`
	Error     *string   `json:"error,omitempty"`
	Message   *string   `json:"message,omitempty"`
	RequestID string    `json:"requestId"`
	Timestamp time.Time `json:"timestamp"`
}

// Tenant represents a tenant in the system
type Tenant struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	DisplayName   string                 `json:"displayName"`
	Status        TenantStatus           `json:"status"`
	Tier          TenantTier             `json:"tier"`
	CreatedAt     time.Time              `json:"createdAt"`
	UpdatedAt     time.Time              `json:"updatedAt"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	ResourceUsage map[string]interface{} `json:"resourceUsage,omitempty"`
}

// TenantTemplate represents a tenant template
type TenantTemplate struct {
	ID                   string                 `json:"id"`
	Name                 string                 `json:"name"`
	DisplayName          string                 `json:"displayName"`
	Description          string                 `json:"description"`
	Category             string                 `json:"category"`
	Version              string                 `json:"version"`
	Configuration        map[string]interface{} `json:"configuration"`
	ResourceRequirements map[string]interface{} `json:"resourceRequirements"`
	CreatedAt            time.Time              `json:"createdAt"`
	IsPublic             bool                   `json:"isPublic"`
	Tags                 []string               `json:"tags"`
}

// CreateTenantRequest represents a request to create a tenant
type CreateTenantRequest struct {
	Name          string                 `json:"name"`
	Tier          TenantTier             `json:"tier"`
	DisplayName   *string                `json:"displayName,omitempty"`
	TemplateID    *string                `json:"templateId,omitempty"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// UpdateTenantRequest represents a request to update a tenant
type UpdateTenantRequest struct {
	DisplayName   *string                `json:"displayName,omitempty"`
	Tier          *TenantTier            `json:"tier,omitempty"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// DeploymentResult represents the result of a deployment
type DeploymentResult struct {
	ID                string           `json:"id"`
	TenantID          string           `json:"tenantId"`
	Status            DeploymentStatus `json:"status"`
	Strategy          string           `json:"strategy"`
	Version           string           `json:"version"`
	StartedAt         time.Time        `json:"startedAt"`
	CompletedAt       *time.Time       `json:"completedAt,omitempty"`
	Logs              []string         `json:"logs"`
	RollbackAvailable bool             `json:"rollbackAvailable"`
}

// AnalyticsMetrics represents tenant analytics metrics
type AnalyticsMetrics struct {
	TenantID         string    `json:"tenantId"`
	Timestamp        time.Time `json:"timestamp"`
	CPUUsagePercent  float64   `json:"cpuUsagePercent"`
	MemoryUsageMB    float64   `json:"memoryUsageMb"`
	StorageUsageGB   float64   `json:"storageUsageGb"`
	RequestCount     int       `json:"requestCount"`
	ErrorRate        float64   `json:"errorRate"`
	ResponseTimeMS   float64   `json:"responseTimeMs"`
	ActiveUsers      int       `json:"activeUsers"`
}

// Request Options

// ListTenantsOptions represents options for listing tenants
type ListTenantsOptions struct {
	Status *TenantStatus `json:"status,omitempty"`
	Tier   *TenantTier   `json:"tier,omitempty"`
	Limit  int           `json:"limit,omitempty"`
	Offset int           `json:"offset,omitempty"`
}

// ListTemplatesOptions represents options for listing templates
type ListTemplatesOptions struct {
	Category   *string `json:"category,omitempty"`
	PublicOnly bool    `json:"publicOnly,omitempty"`
	Limit      int     `json:"limit,omitempty"`
}

// GetMetricsOptions represents options for getting metrics
type GetMetricsOptions struct {
	StartTime *time.Time `json:"startTime,omitempty"`
	EndTime   *time.Time `json:"endTime,omitempty"`
	Interval  string     `json:"interval,omitempty"`
}

// Custom Errors

// MTenError represents a generic MTen SDK error
type MTenError struct {
	Message      string
	StatusCode   *int
	ResponseData map[string]interface{}
}

func (e *MTenError) Error() string {
	if e.StatusCode != nil {
		return fmt.Sprintf("MTen API error (HTTP %d): %s", *e.StatusCode, e.Message)
	}
	return fmt.Sprintf("MTen SDK error: %s", e.Message)
}

// AuthenticationError represents authentication-related errors
type AuthenticationError struct {
	*MTenError
}

// ValidationError represents validation-related errors
type ValidationError struct {
	*MTenError
}

// NetworkError represents network-related errors
type NetworkError struct {
	*MTenError
}

// Client Configuration

// ClientOptions represents configuration options for the MTen client
type ClientOptions struct {
	Timeout       time.Duration
	RetryAttempts int
	RetryDelay    time.Duration
	UserAgent     string
	HTTPClient    *http.Client
}

// DefaultClientOptions returns default client options
func DefaultClientOptions() *ClientOptions {
	return &ClientOptions{
		Timeout:       30 * time.Second,
		RetryAttempts: 3,
		RetryDelay:    1 * time.Second,
		UserAgent:     fmt.Sprintf("MTen-Go-SDK/%s", Version),
		HTTPClient:    &http.Client{},
	}
}

// MTen Client

// Client represents the MTen API client
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	options    *ClientOptions
}

// NewClient creates a new MTen API client
func NewClient(baseURL, apiKey string, options *ClientOptions) *Client {
	if options == nil {
		options = DefaultClientOptions()
	}

	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{
			Timeout: options.Timeout,
		}
	}

	return &Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: options.HTTPClient,
		options:    options,
	}
}

// Helper method to make HTTP requests with retry logic
func (c *Client) request(ctx context.Context, method, endpoint string, data interface{}, params map[string]string) ([]byte, error) {
	u, err := url.Parse(c.baseURL + "/api/v1" + endpoint)
	if err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("invalid URL: %v", err)}
	}

	if params != nil {
		q := u.Query()
		for k, v := range params {
			q.Set(k, v)
		}
		u.RawQuery = q.Encode()
	}

	var body io.Reader
	if data != nil {
		jsonData, err := json.Marshal(data)
		if err != nil {
			return nil, &MTenError{Message: fmt.Sprintf("failed to marshal request data: %v", err)}
		}
		body = bytes.NewReader(jsonData)
	}

	var lastErr error
	for attempt := 0; attempt <= c.options.RetryAttempts; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, u.String(), body)
		if err != nil {
			return nil, &MTenError{Message: fmt.Sprintf("failed to create request: %v", err)}
		}

		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("User-Agent", c.options.UserAgent)

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = &NetworkError{MTenError: &MTenError{Message: fmt.Sprintf("request failed: %v", err)}}
			if attempt < c.options.RetryAttempts {
				time.Sleep(c.options.RetryDelay * time.Duration(1<<attempt)) // Exponential backoff
				continue
			}
			return nil, lastErr
		}
		defer resp.Body.Close()

		responseBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, &MTenError{Message: fmt.Sprintf("failed to read response: %v", err)}
		}

		if resp.StatusCode >= 400 {
			var errorData map[string]interface{}
			json.Unmarshal(responseBody, &errorData)

			message := "Unknown error"
			if msg, ok := errorData["message"].(string); ok {
				message = msg
			}

			baseError := &MTenError{
				Message:      message,
				StatusCode:   &resp.StatusCode,
				ResponseData: errorData,
			}

			switch resp.StatusCode {
			case 401:
				return nil, &AuthenticationError{MTenError: baseError}
			case 422:
				return nil, &ValidationError{MTenError: baseError}
			default:
				return nil, baseError
			}
		}

		return responseBody, nil
	}

	return nil, lastErr
}

// Tenant Management Methods

// ListTenants lists tenants with optional filtering
func (c *Client) ListTenants(ctx context.Context, options *ListTenantsOptions) (*APIResponse[[]Tenant], error) {
	params := make(map[string]string)

	if options != nil {
		if options.Status != nil {
			params["status"] = string(*options.Status)
		}
		if options.Tier != nil {
			params["tier"] = string(*options.Tier)
		}
		if options.Limit > 0 {
			params["limit"] = strconv.Itoa(options.Limit)
		} else {
			params["limit"] = "100"
		}
		if options.Offset > 0 {
			params["offset"] = strconv.Itoa(options.Offset)
		}
	} else {
		params["limit"] = "100"
		params["offset"] = "0"
	}

	data, err := c.request(ctx, "GET", "/tenants", nil, params)
	if err != nil {
		return nil, err
	}

	var response APIResponse[[]Tenant]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// GetTenant gets a tenant by ID
func (c *Client) GetTenant(ctx context.Context, tenantID string) (*APIResponse[Tenant], error) {
	data, err := c.request(ctx, "GET", "/tenants/"+tenantID, nil, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[Tenant]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// CreateTenant creates a new tenant
func (c *Client) CreateTenant(ctx context.Context, req *CreateTenantRequest) (*APIResponse[Tenant], error) {
	data, err := c.request(ctx, "POST", "/tenants", req, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[Tenant]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// UpdateTenant updates an existing tenant
func (c *Client) UpdateTenant(ctx context.Context, tenantID string, req *UpdateTenantRequest) (*APIResponse[Tenant], error) {
	data, err := c.request(ctx, "PATCH", "/tenants/"+tenantID, req, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[Tenant]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// DeleteTenant deletes a tenant
func (c *Client) DeleteTenant(ctx context.Context, tenantID string, force bool) (*APIResponse[bool], error) {
	params := make(map[string]string)
	if force {
		params["force"] = "true"
	}

	data, err := c.request(ctx, "DELETE", "/tenants/"+tenantID, nil, params)
	if err != nil {
		return nil, err
	}

	var response APIResponse[bool]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// Template Management Methods

// ListTemplates lists available tenant templates
func (c *Client) ListTemplates(ctx context.Context, options *ListTemplatesOptions) (*APIResponse[[]TenantTemplate], error) {
	params := make(map[string]string)

	if options != nil {
		if options.Category != nil {
			params["category"] = *options.Category
		}
		params["publicOnly"] = strconv.FormatBool(options.PublicOnly)
		if options.Limit > 0 {
			params["limit"] = strconv.Itoa(options.Limit)
		} else {
			params["limit"] = "50"
		}
	} else {
		params["limit"] = "50"
		params["publicOnly"] = "true"
	}

	data, err := c.request(ctx, "GET", "/templates", nil, params)
	if err != nil {
		return nil, err
	}

	var response APIResponse[[]TenantTemplate]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// GetTemplate gets a template by ID
func (c *Client) GetTemplate(ctx context.Context, templateID string) (*APIResponse[TenantTemplate], error) {
	data, err := c.request(ctx, "GET", "/templates/"+templateID, nil, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[TenantTemplate]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// CreateTemplate creates a new tenant template
func (c *Client) CreateTemplate(ctx context.Context, template *TenantTemplate) (*APIResponse[TenantTemplate], error) {
	data, err := c.request(ctx, "POST", "/templates", template, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[TenantTemplate]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// Deployment Methods

// DeployTenant deploys a tenant with the specified strategy
func (c *Client) DeployTenant(ctx context.Context, tenantID string, version *string, strategy string) (*APIResponse[DeploymentResult], error) {
	req := map[string]interface{}{
		"tenantId": tenantID,
		"strategy": strategy,
	}

	if version != nil {
		req["version"] = *version
	}

	data, err := c.request(ctx, "POST", "/deployments", req, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[DeploymentResult]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// GetDeploymentStatus gets deployment status
func (c *Client) GetDeploymentStatus(ctx context.Context, deploymentID string) (*APIResponse[DeploymentResult], error) {
	data, err := c.request(ctx, "GET", "/deployments/"+deploymentID, nil, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[DeploymentResult]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// RollbackDeployment rolls back a deployment to a previous version
func (c *Client) RollbackDeployment(ctx context.Context, deploymentID string, targetVersion *string) (*APIResponse[DeploymentResult], error) {
	req := make(map[string]interface{})
	if targetVersion != nil {
		req["targetVersion"] = *targetVersion
	}

	data, err := c.request(ctx, "POST", "/deployments/"+deploymentID+"/rollback", req, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[DeploymentResult]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// Analytics Methods

// GetTenantMetrics gets tenant analytics metrics
func (c *Client) GetTenantMetrics(ctx context.Context, tenantID string, options *GetMetricsOptions) (*APIResponse[[]AnalyticsMetrics], error) {
	params := map[string]string{
		"interval": "1h",
	}

	if options != nil {
		if options.StartTime != nil {
			params["startTime"] = options.StartTime.Format(time.RFC3339)
		}
		if options.EndTime != nil {
			params["endTime"] = options.EndTime.Format(time.RFC3339)
		}
		if options.Interval != "" {
			params["interval"] = options.Interval
		}
	}

	data, err := c.request(ctx, "GET", "/tenants/"+tenantID+"/metrics", nil, params)
	if err != nil {
		return nil, err
	}

	var response APIResponse[[]AnalyticsMetrics]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// GetTenantHealthScore gets tenant health score
func (c *Client) GetTenantHealthScore(ctx context.Context, tenantID string) (*APIResponse[float64], error) {
	data, err := c.request(ctx, "GET", "/tenants/"+tenantID+"/health", nil, nil)
	if err != nil {
		return nil, err
	}

	var rawResponse APIResponse[map[string]interface{}]
	if err := json.Unmarshal(data, &rawResponse); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	var healthScore float64
	if rawResponse.Data != nil {
		if score, ok := (*rawResponse.Data)["healthScore"].(float64); ok {
			healthScore = score
		}
	}

	return &APIResponse[float64]{
		Success:   rawResponse.Success,
		Data:      &healthScore,
		Error:     rawResponse.Error,
		Message:   rawResponse.Message,
		RequestID: rawResponse.RequestID,
		Timestamp: rawResponse.Timestamp,
	}, nil
}

// Utility Methods

// Ping pings the API to check connectivity
func (c *Client) Ping(ctx context.Context) (*APIResponse[map[string]interface{}], error) {
	data, err := c.request(ctx, "GET", "/ping", nil, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[map[string]interface{}]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// GetAPIInfo gets API version and capability information
func (c *Client) GetAPIInfo(ctx context.Context) (*APIResponse[map[string]interface{}], error) {
	data, err := c.request(ctx, "GET", "/info", nil, nil)
	if err != nil {
		return nil, err
	}

	var response APIResponse[map[string]interface{}]
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, &MTenError{Message: fmt.Sprintf("failed to unmarshal response: %v", err)}
	}

	return &response, nil
}

// High-level convenience functions

// QuickTenantSetup creates a tenant with sensible defaults and optional template
func QuickTenantSetup(ctx context.Context, client *Client, name string, tier TenantTier, templateName *string) (*Tenant, error) {
	var templateID *string

	if templateName != nil {
		templatesResp, err := client.ListTemplates(ctx, &ListTemplatesOptions{
			PublicOnly: true,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list templates: %w", err)
		}

		if templatesResp.Success && templatesResp.Data != nil {
			for _, template := range *templatesResp.Data {
				if template.Name == *templateName {
					templateID = &template.ID
					break
				}
			}
		}
	}

	tenantResp, err := client.CreateTenant(ctx, &CreateTenantRequest{
		Name:       name,
		Tier:       tier,
		TemplateID: templateID,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create tenant: %w", err)
	}

	if !tenantResp.Success || tenantResp.Data == nil {
		errorMsg := "unknown error"
		if tenantResp.Error != nil {
			errorMsg = *tenantResp.Error
		}
		return nil, fmt.Errorf("tenant creation failed: %s", errorMsg)
	}

	return tenantResp.Data, nil
}