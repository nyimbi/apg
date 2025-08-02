// APG Central Configuration - API Load Testing with k6
// Comprehensive load testing scenarios for the API

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const apiErrors = new Counter('api_errors');
const apiSuccessRate = new Rate('api_success_rate');
const configurationCreationTime = new Trend('configuration_creation_time');
const searchResponseTime = new Trend('search_response_time');
const aiOptimizationTime = new Trend('ai_optimization_time');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'cc_test_api_key_2025';

// Test options
export const options = {
	scenarios: {
		// Smoke test - minimal load
		smoke_test: {
			executor: 'constant-vus',
			vus: 1,
			duration: '1m',
			tags: { test_type: 'smoke' },
			env: { SCENARIO: 'smoke' }
		},
		
		// Load test - normal expected load
		load_test: {
			executor: 'ramping-vus',
			startVUs: 0,
			stages: [
				{ duration: '2m', target: 10 },  // Ramp up
				{ duration: '5m', target: 10 },  // Stay at 10 users
				{ duration: '2m', target: 20 },  // Ramp to 20 users
				{ duration: '5m', target: 20 },  // Stay at 20 users
				{ duration: '2m', target: 0 },   // Ramp down
			],
			tags: { test_type: 'load' },
			env: { SCENARIO: 'load' }
		},
		
		// Stress test - above normal load
		stress_test: {
			executor: 'ramping-vus',
			startVUs: 0,
			stages: [
				{ duration: '2m', target: 20 },  // Ramp up
				{ duration: '8m', target: 50 },  // Ramp to 50 users
				{ duration: '2m', target: 0 },   // Ramp down
			],
			tags: { test_type: 'stress' },
			env: { SCENARIO: 'stress' }
		},
		
		// Spike test - sudden load increase
		spike_test: {
			executor: 'ramping-vus',
			startVUs: 0,
			stages: [
				{ duration: '1m', target: 10 },   // Normal load
				{ duration: '30s', target: 100 }, // Spike to 100 users
				{ duration: '3m', target: 100 },  // Stay at spike
				{ duration: '30s', target: 10 },  // Drop to normal
				{ duration: '1m', target: 0 },    // Ramp down
			],
			tags: { test_type: 'spike' },
			env: { SCENARIO: 'spike' }
		},
		
		// Volume test - large amount of data
		volume_test: {
			executor: 'constant-vus',
			vus: 5,
			duration: '10m',
			tags: { test_type: 'volume' },
			env: { SCENARIO: 'volume' }
		}
	},
	
	thresholds: {
		// Overall thresholds
		http_req_duration: ['p(95)<500', 'p(99)<1000'],
		http_req_failed: ['rate<0.05'], // Less than 5% failure rate
		
		// API-specific thresholds
		'http_req_duration{group:::API Health Check}': ['p(95)<100'],
		'http_req_duration{group:::Configuration Management}': ['p(95)<800'],
		'http_req_duration{group:::AI Features}': ['p(95)<2000'],
		'http_req_duration{group:::Search Operations}': ['p(95)<1000'],
		
		// Custom metric thresholds
		api_success_rate: ['rate>0.95'],
		configuration_creation_time: ['p(95)<1000'],
		search_response_time: ['p(95)<800'],
		ai_optimization_time: ['p(95)<3000'],
	}
};

// Test data
const testWorkspaceId = 'test_workspace_' + Date.now();
const configurationNames = [
	'database-config', 'redis-config', 'api-config', 'cache-config',
	'security-config', 'monitoring-config', 'logging-config', 'network-config'
];

// Helper functions
function getAuthHeaders() {
	return {
		'Content-Type': 'application/json',
		'X-API-Key': API_KEY,
		'Accept': 'application/json'
	};
}

function generateConfigurationData(name) {
	return {
		name: `${name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
		key_path: `/app/${name.replace('-config', '')}`,
		value: {
			host: 'localhost',
			port: Math.floor(Math.random() * 9000) + 1000,
			timeout: Math.floor(Math.random() * 300) + 30,
			max_connections: Math.floor(Math.random() * 100) + 10,
			enabled: true,
			features: {
				logging: true,
				monitoring: true,
				caching: Math.random() > 0.5
			}
		},
		security_level: ['public', 'internal', 'confidential'][Math.floor(Math.random() * 3)],
		tags: ['load-test', 'automated', name.split('-')[0]]
	};
}

// Setup function
export function setup() {
	console.log('🚀 Starting APG Central Configuration Load Tests');
	console.log(`Target: ${BASE_URL}`);
	
	// Health check
	const healthResponse = http.get(`${BASE_URL}/health`);
	if (healthResponse.status !== 200) {
		throw new Error(`Health check failed: ${healthResponse.status}`);
	}
	
	console.log('✅ Health check passed');
	
	return {
		startTime: Date.now(),
		baseUrl: BASE_URL
	};
}

// Main test function
export default function(data) {
	const scenario = __ENV.SCENARIO || 'load';
	
	group('API Health Check', () => {
		const response = http.get(`${BASE_URL}/health`, {
			headers: getAuthHeaders()
		});
		
		check(response, {
			'health check status is 200': (r) => r.status === 200,
			'health check response time < 100ms': (r) => r.timings.duration < 100,
			'health check has correct status': (r) => JSON.parse(r.body).status === 'healthy'
		});
		
		if (response.status !== 200) {
			apiErrors.add(1);
		}
		apiSuccessRate.add(response.status === 200);
	});
	
	group('Configuration Management', () => {
		// Create configuration
		const configName = configurationNames[Math.floor(Math.random() * configurationNames.length)];
		const configData = generateConfigurationData(configName);
		
		const createResponse = http.post(`${BASE_URL}/configurations`, JSON.stringify(configData), {
			headers: getAuthHeaders(),
			params: { workspace_id: testWorkspaceId }
		});
		
		const createSuccess = check(createResponse, {
			'create configuration status is 200': (r) => r.status === 200 || r.status === 201,
			'create configuration response time < 1s': (r) => r.timings.duration < 1000,
			'create configuration returns valid response': (r) => {
				try {
					const body = JSON.parse(r.body);
					return body.id && body.name;
				} catch (e) {
					return false;
				}
			}
		});
		
		if (createSuccess) {
			configurationCreationTime.add(createResponse.timings.duration);
			
			const responseBody = JSON.parse(createResponse.body);
			const configId = responseBody.id;
			
			// Get configuration
			const getResponse = http.get(`${BASE_URL}/configurations/${configId}`, {
				headers: getAuthHeaders()
			});
			
			check(getResponse, {
				'get configuration status is 200': (r) => r.status === 200,
				'get configuration response time < 500ms': (r) => r.timings.duration < 500,
				'get configuration returns correct data': (r) => {
					try {
						const body = JSON.parse(r.body);
						return body.id === configId && body.name === configData.name;
					} catch (e) {
						return false;
					}
				}
			});
			
			// Update configuration (occasionally)
			if (Math.random() < 0.3) {
				const updateData = {
					value: {
						...configData.value,
						updated_at: new Date().toISOString(),
						version: '1.1.0'
					}
				};
				
				const updateResponse = http.put(`${BASE_URL}/configurations/${configId}`, JSON.stringify(updateData), {
					headers: getAuthHeaders()
				});
				
				check(updateResponse, {
					'update configuration status is 200': (r) => r.status === 200,
					'update configuration response time < 800ms': (r) => r.timings.duration < 800
				});
			}
		} else {
			apiErrors.add(1);
		}
		
		apiSuccessRate.add(createSuccess);
	});
	
	group('Search Operations', () => {
		const searchQueries = [
			'database',
			'redis',
			'config',
			'load-test',
			'production'
		];
		
		const query = searchQueries[Math.floor(Math.random() * searchQueries.length)];
		const searchResponse = http.get(`${BASE_URL}/configurations?query=${query}&limit=20`, {
			headers: getAuthHeaders()
		});
		
		const searchSuccess = check(searchResponse, {
			'search status is 200': (r) => r.status === 200,
			'search response time < 1s': (r) => r.timings.duration < 1000,
			'search returns results': (r) => {
				try {
					const body = JSON.parse(r.body);
					return body.configurations && Array.isArray(body.configurations);
				} catch (e) {
					return false;
				}
			}
		});
		
		if (searchSuccess) {
			searchResponseTime.add(searchResponse.timings.duration);
		} else {
			apiErrors.add(1);
		}
		
		apiSuccessRate.add(searchSuccess);
	});
	
	// AI Features (less frequent to avoid overloading AI service)
	if (Math.random() < 0.2) {
		group('AI Features', () => {
			// Natural language query
			const nlQueries = [
				'find all database configurations',
				'show me redis cache settings',
				'configurations with high security level',
				'recent configuration changes'
			];
			
			const nlQuery = nlQueries[Math.floor(Math.random() * nlQueries.length)];
			const nlResponse = http.post(`${BASE_URL}/configurations/natural-language-query`, JSON.stringify({
				query: nlQuery
			}), {
				headers: getAuthHeaders(),
				timeout: '10s'
			});
			
			const nlSuccess = check(nlResponse, {
				'NL query status is 200 or 503': (r) => r.status === 200 || r.status === 503, // 503 if AI unavailable
				'NL query response time < 5s': (r) => r.timings.duration < 5000
			});
			
			if (nlResponse.status === 200) {
				// Test AI optimization (even less frequent)
				if (Math.random() < 0.1) {
					const optimizeResponse = http.post(`${BASE_URL}/configurations/test-config-123/optimize`, '', {
						headers: getAuthHeaders(),
						timeout: '15s'
					});
					
					if (optimizeResponse.status === 200) {
						aiOptimizationTime.add(optimizeResponse.timings.duration);
					}
					
					check(optimizeResponse, {
						'AI optimization completed or service unavailable': (r) => r.status === 200 || r.status === 404 || r.status === 503,
						'AI optimization response time < 10s': (r) => r.timings.duration < 10000
					});
				}
			}
			
			apiSuccessRate.add(nlSuccess);
			if (!nlSuccess) {
				apiErrors.add(1);
			}
		});
	}
	
	// Vary sleep time based on scenario
	let sleepTime = 1;
	switch (scenario) {
		case 'smoke':
			sleepTime = 2;
			break;
		case 'load':
			sleepTime = Math.random() * 2 + 0.5; // 0.5-2.5s
			break;
		case 'stress':
			sleepTime = Math.random() * 1 + 0.2; // 0.2-1.2s
			break;
		case 'spike':
			sleepTime = Math.random() * 0.5; // 0-0.5s
			break;
		case 'volume':
			sleepTime = 0.1; // Minimal sleep for volume test
			break;
	}
	
	sleep(sleepTime);
}

// Teardown function
export function teardown(data) {
	console.log('🏁 Load test completed');
	console.log(`Duration: ${((Date.now() - data.startTime) / 1000).toFixed(2)}s`);
	
	// Cleanup health check
	const healthResponse = http.get(`${BASE_URL}/health`);
	console.log(healthResponse.status === 200 ? '✅ System healthy after test' : '⚠️ System may need attention');
}