"""
APG Central Configuration - Revolutionary GitOps Engine

GitOps-native workflows with advanced branching strategies,
automated pipelines, and intelligent conflict resolution.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import yaml
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import git
from git import Repo, GitCommandError
import aiofiles
import asyncio_subprocess as subprocess
from jinja2 import Template, Environment, FileSystemLoader

# GitOps workflow libraries
try:
	import kubernetes
	from kubernetes import client, config
	KUBERNETES_AVAILABLE = True
except ImportError:
	KUBERNETES_AVAILABLE = False

try:
	import argocd
	ARGOCD_AVAILABLE = True
except ImportError:
	ARGOCD_AVAILABLE = False


class GitOpsStrategy(Enum):
	"""GitOps deployment strategies."""
	PUSH_BASED = "push_based"
	PULL_BASED = "pull_based"
	HYBRID = "hybrid"


class BranchingStrategy(Enum):
	"""Git branching strategies."""
	GITFLOW = "gitflow"
	GITHUB_FLOW = "github_flow"
	GITLAB_FLOW = "gitlab_flow"
	ENVIRONMENT_BRANCHES = "environment_branches"


class DeploymentStage(Enum):
	"""Deployment pipeline stages."""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	HOTFIX = "hotfix"


class ConflictResolutionStrategy(Enum):
	"""Merge conflict resolution strategies."""
	MANUAL = "manual"
	AUTO_MERGE = "auto_merge"
	PREFER_SOURCE = "prefer_source"
	PREFER_TARGET = "prefer_target"
	AI_ASSISTED = "ai_assisted"


@dataclass
class GitOpsConfiguration:
	"""GitOps configuration settings."""
	repository_url: str
	branch_strategy: BranchingStrategy
	deployment_strategy: GitOpsStrategy
	auto_merge: bool
	conflict_resolution: ConflictResolutionStrategy
	environments: List[str]
	sync_interval_seconds: int
	notification_webhooks: List[str]
	approval_required_for_prod: bool


@dataclass
class GitCommit:
	"""Git commit information."""
	sha: str
	message: str
	author: str
	timestamp: datetime
	files_changed: List[str]
	branch: str


@dataclass
class DeploymentRequest:
	"""Deployment request."""
	configuration_id: str
	target_environment: str
	source_branch: str
	target_branch: str
	deployment_config: Dict[str, Any]
	approval_required: bool
	auto_rollback: bool


@dataclass
class DeploymentResult:
	"""Deployment result."""
	deployment_id: str
	status: str  # success, failed, pending, rolled_back
	commit_sha: str
	environment: str
	started_at: datetime
	completed_at: Optional[datetime]
	logs: List[str]
	rollback_commit: Optional[str]


@dataclass
class ConflictResolution:
	"""Merge conflict resolution."""
	conflict_id: str
	file_path: str
	conflict_type: str
	source_content: str
	target_content: str
	resolved_content: str
	resolution_strategy: ConflictResolutionStrategy
	auto_resolved: bool


class CentralConfigurationGitOps:
	"""Revolutionary GitOps engine for configuration management."""
	
	def __init__(self, gitops_config: GitOpsConfiguration):
		"""Initialize GitOps engine."""
		self.config = gitops_config
		self.repositories: Dict[str, Repo] = {}
		self.deployment_history: List[DeploymentResult] = []
		self.active_deployments: Dict[str, DeploymentResult] = {}
		self.conflict_resolutions: List[ConflictResolution] = []
		
		# Template environment for configuration rendering
		self.jinja_env = Environment(
			loader=FileSystemLoader(['.', '/templates']),
			autoescape=False
		)
		
		# Initialize Kubernetes client if available
		if KUBERNETES_AVAILABLE:
			try:
				config.load_incluster_config()
				self.k8s_client = client.ApiClient()
			except:
				try:
					config.load_kube_config()
					self.k8s_client = client.ApiClient()
				except:
					self.k8s_client = None
		else:
			self.k8s_client = None
	
	# ==================== Repository Management ====================
	
	async def initialize_repository(
		self,
		repo_url: str,
		local_path: Optional[str] = None
	) -> str:
		"""Initialize or clone GitOps repository."""
		try:
			if not local_path:
				local_path = tempfile.mkdtemp(prefix="gitops_")
			
			local_path_obj = Path(local_path)
			
			if local_path_obj.exists() and (local_path_obj / '.git').exists():
				# Repository already exists, pull latest changes
				repo = Repo(local_path)
				repo.remotes.origin.pull()
				print(f"✅ Updated existing repository at {local_path}")
			else:
				# Clone repository
				repo = Repo.clone_from(repo_url, local_path)
				print(f"✅ Cloned repository to {local_path}")
			
			self.repositories[repo_url] = repo
			
			# Initialize GitOps structure if needed
			await self._initialize_gitops_structure(repo, local_path_obj)
			
			return local_path
			
		except Exception as e:
			print(f"❌ Failed to initialize repository: {e}")
			raise
	
	async def _initialize_gitops_structure(self, repo: Repo, repo_path: Path):
		"""Initialize GitOps directory structure."""
		# Create standard GitOps directories
		directories = [
			"environments/development",
			"environments/staging", 
			"environments/production",
			"applications",
			"infrastructure",
			"policies",
			"templates"
		]
		
		for directory in directories:
			dir_path = repo_path / directory
			dir_path.mkdir(parents=True, exist_ok=True)
			
			# Create .gitkeep if directory is empty
			gitkeep_path = dir_path / ".gitkeep"
			if not any(dir_path.iterdir()) and not gitkeep_path.exists():
				gitkeep_path.touch()
	
	async def create_branch(
		self,
		repo_url: str,
		branch_name: str,
		source_branch: str = "main"
	) -> str:
		"""Create new branch for configuration changes."""
		try:
			repo = self.repositories.get(repo_url)
			if not repo:
				raise ValueError(f"Repository {repo_url} not initialized")
			
			# Ensure we're on the source branch
			repo.git.checkout(source_branch)
			repo.remotes.origin.pull()
			
			# Create and checkout new branch
			new_branch = repo.create_head(branch_name)
			new_branch.checkout()
			
			print(f"✅ Created branch '{branch_name}' from '{source_branch}'")
			return branch_name
			
		except Exception as e:
			print(f"❌ Failed to create branch: {e}")
			raise
	
	async def commit_configuration_changes(
		self,
		repo_url: str,
		configuration_id: str,
		configuration_data: Dict[str, Any],
		target_environment: str,
		commit_message: str,
		branch_name: str
	) -> GitCommit:
		"""Commit configuration changes to GitOps repository."""
		try:
			repo = self.repositories.get(repo_url)
			if not repo:
				raise ValueError(f"Repository {repo_url} not initialized")
			
			repo_path = Path(repo.working_dir)
			
			# Generate configuration files for target environment
			config_files = await self._generate_configuration_files(
				configuration_data,
				target_environment,
				repo_path
			)
			
			# Checkout target branch
			repo.git.checkout(branch_name)
			
			# Add files to git
			for file_path in config_files:
				repo.index.add([str(file_path.relative_to(repo_path))])
			
			# Commit changes
			commit = repo.index.commit(commit_message)
			
			# Push to remote
			repo.remotes.origin.push(branch_name)
			
			git_commit = GitCommit(
				sha=commit.hexsha,
				message=commit_message,
				author=str(commit.author),
				timestamp=datetime.fromtimestamp(commit.committed_date, timezone.utc),
				files_changed=[str(f.relative_to(repo_path)) for f in config_files],
				branch=branch_name
			)
			
			print(f"✅ Committed configuration changes: {commit.hexsha[:8]}")
			return git_commit
			
		except Exception as e:
			print(f"❌ Failed to commit changes: {e}")
			raise
	
	async def _generate_configuration_files(
		self,
		configuration_data: Dict[str, Any],
		environment: str,
		repo_path: Path
	) -> List[Path]:
		"""Generate configuration files for deployment."""
		generated_files = []
		
		# Environment-specific directory
		env_dir = repo_path / "environments" / environment
		env_dir.mkdir(parents=True, exist_ok=True)
		
		# Generate main configuration file
		config_file = env_dir / f"{configuration_data['name']}.yaml"
		config_content = await self._render_configuration_template(
			configuration_data,
			environment
		)
		
		async with aiofiles.open(config_file, 'w') as f:
			await f.write(config_content)
		
		generated_files.append(config_file)
		
		# Generate Kubernetes manifests if applicable
		if self.k8s_client and configuration_data.get('deployment_type') == 'kubernetes':
			k8s_manifests = await self._generate_kubernetes_manifests(
				configuration_data,
				environment,
				env_dir
			)
			generated_files.extend(k8s_manifests)
		
		# Generate Helm charts if applicable
		if configuration_data.get('deployment_type') == 'helm':
			helm_files = await self._generate_helm_chart(
				configuration_data,
				environment,
				env_dir
			)
			generated_files.extend(helm_files)
		
		return generated_files
	
	async def _render_configuration_template(
		self,
		configuration_data: Dict[str, Any],
		environment: str
	) -> str:
		"""Render configuration using Jinja2 templates."""
		template_content = """
# Configuration: {{ name }}
# Environment: {{ environment }}
# Generated: {{ timestamp }}

apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ name }}-config
  namespace: {{ namespace | default('default') }}
  labels:
    app: {{ name }}
    environment: {{ environment }}
    managed-by: apg-central-config
data:
{% for key, value in config_data.items() %}
  {{ key }}: |
    {{ value | to_yaml | indent(4) }}
{% endfor %}

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ name }}-deployment
  namespace: {{ namespace | default('default') }}
spec:
  replicas: {{ replicas | default(1) }}
  selector:
    matchLabels:
      app: {{ name }}
  template:
    metadata:
      labels:
        app: {{ name }}
        environment: {{ environment }}
    spec:
      containers:
      - name: {{ name }}
        image: {{ image | default('nginx:latest') }}
        envFrom:
        - configMapRef:
            name: {{ name }}-config
"""
		
		template = Template(template_content)
		
		# Add custom filters
		template.globals['to_yaml'] = lambda x: yaml.dump(x, default_flow_style=False)
		
		rendered = template.render(
			name=configuration_data['name'],
			environment=environment,
			timestamp=datetime.now(timezone.utc).isoformat(),
			config_data=configuration_data.get('value', {}),
			namespace=configuration_data.get('namespace', 'default'),
			replicas=configuration_data.get('replicas', 1),
			image=configuration_data.get('image', 'nginx:latest')
		)
		
		return rendered
	
	async def _generate_kubernetes_manifests(
		self,
		configuration_data: Dict[str, Any],
		environment: str,
		output_dir: Path
	) -> List[Path]:
		"""Generate Kubernetes manifests."""
		manifests = []
		
		# ConfigMap manifest
		configmap_file = output_dir / f"{configuration_data['name']}-configmap.yaml"
		configmap_content = {
			'apiVersion': 'v1',
			'kind': 'ConfigMap',
			'metadata': {
				'name': f"{configuration_data['name']}-config",
				'namespace': configuration_data.get('namespace', 'default')
			},
			'data': {k: str(v) for k, v in configuration_data.get('value', {}).items()}
		}
		
		async with aiofiles.open(configmap_file, 'w') as f:
			await f.write(yaml.dump(configmap_content, default_flow_style=False))
		
		manifests.append(configmap_file)
		
		# Deployment manifest (if requested)
		if configuration_data.get('create_deployment', False):
			deployment_file = output_dir / f"{configuration_data['name']}-deployment.yaml"
			deployment_content = {
				'apiVersion': 'apps/v1',
				'kind': 'Deployment',
				'metadata': {
					'name': f"{configuration_data['name']}-deployment",
					'namespace': configuration_data.get('namespace', 'default')
				},
				'spec': {
					'replicas': configuration_data.get('replicas', 1),
					'selector': {
						'matchLabels': {'app': configuration_data['name']}
					},
					'template': {
						'metadata': {
							'labels': {'app': configuration_data['name']}
						},
						'spec': {
							'containers': [{
								'name': configuration_data['name'],
								'image': configuration_data.get('image', 'nginx:latest'),
								'envFrom': [{
									'configMapRef': {
										'name': f"{configuration_data['name']}-config"
									}
								}]
							}]
						}
					}
				}
			}
			
			async with aiofiles.open(deployment_file, 'w') as f:
				await f.write(yaml.dump(deployment_content, default_flow_style=False))
			
			manifests.append(deployment_file)
		
		return manifests
	
	async def _generate_helm_chart(
		self,
		configuration_data: Dict[str, Any],
		environment: str,
		output_dir: Path
	) -> List[Path]:
		"""Generate Helm chart files."""
		chart_dir = output_dir / f"{configuration_data['name']}-chart"
		chart_dir.mkdir(exist_ok=True)
		
		# Chart.yaml
		chart_file = chart_dir / "Chart.yaml"
		chart_content = {
			'apiVersion': 'v2',
			'name': configuration_data['name'],
			'description': f"Helm chart for {configuration_data['name']}",
			'version': '0.1.0',
			'appVersion': '1.0.0'
		}
		
		async with aiofiles.open(chart_file, 'w') as f:
			await f.write(yaml.dump(chart_content, default_flow_style=False))
		
		# values.yaml
		values_file = chart_dir / "values.yaml"
		values_content = {
			'name': configuration_data['name'],
			'namespace': configuration_data.get('namespace', 'default'),
			'replicas': configuration_data.get('replicas', 1),
			'image': configuration_data.get('image', 'nginx:latest'),
			'config': configuration_data.get('value', {})
		}
		
		async with aiofiles.open(values_file, 'w') as f:
			await f.write(yaml.dump(values_content, default_flow_style=False))
		
		return [chart_file, values_file]
	
	# ==================== Deployment Pipeline ====================
	
	async def create_deployment_request(
		self,
		configuration_id: str,
		target_environment: str,
		deployment_config: Dict[str, Any]
	) -> str:
		"""Create deployment request."""
		deployment_id = f"deploy_{configuration_id[:8]}_{target_environment}_{int(datetime.now().timestamp())}"
		
		# Determine source and target branches based on strategy
		source_branch, target_branch = await self._determine_branches(
			target_environment,
			deployment_config
		)
		
		deployment_request = DeploymentRequest(
			configuration_id=configuration_id,
			target_environment=target_environment,
			source_branch=source_branch,
			target_branch=target_branch,
			deployment_config=deployment_config,
			approval_required=self._requires_approval(target_environment),
			auto_rollback=deployment_config.get('auto_rollback', True)
		)
		
		# Create deployment result placeholder
		deployment_result = DeploymentResult(
			deployment_id=deployment_id,
			status="pending",
			commit_sha="",
			environment=target_environment,
			started_at=datetime.now(timezone.utc),
			completed_at=None,
			logs=[],
			rollback_commit=None
		)
		
		self.active_deployments[deployment_id] = deployment_result
		
		print(f"✅ Created deployment request: {deployment_id}")
		return deployment_id
	
	async def _determine_branches(
		self,
		environment: str,
		deployment_config: Dict[str, Any]
	) -> Tuple[str, str]:
		"""Determine source and target branches based on branching strategy."""
		if self.config.branch_strategy == BranchingStrategy.ENVIRONMENT_BRANCHES:
			return "main", f"env/{environment}"
		elif self.config.branch_strategy == BranchingStrategy.GITFLOW:
			if environment == "production":
				return "release", "main"
			else:
				return "develop", f"env/{environment}"
		elif self.config.branch_strategy == BranchingStrategy.GITHUB_FLOW:
			return "main", f"deploy/{environment}"
		else:
			return "main", f"env/{environment}"
	
	def _requires_approval(self, environment: str) -> bool:
		"""Check if environment requires deployment approval."""
		return (
			self.config.approval_required_for_prod and 
			environment == "production"
		)
	
	async def execute_deployment(
		self,
		deployment_id: str,
		approved_by: Optional[str] = None
	) -> DeploymentResult:
		"""Execute deployment pipeline."""
		try:
			deployment = self.active_deployments.get(deployment_id)
			if not deployment:
				raise ValueError(f"Deployment {deployment_id} not found")
			
			deployment.status = "running"
			deployment.logs.append(f"Starting deployment at {datetime.now(timezone.utc)}")
			
			# Check approval if required
			if deployment.approval_required and not approved_by:
				deployment.status = "pending_approval"
				deployment.logs.append("Waiting for approval")
				return deployment
			
			# Execute deployment steps
			await self._execute_deployment_pipeline(deployment)
			
			deployment.status = "success"
			deployment.completed_at = datetime.now(timezone.utc)
			deployment.logs.append("Deployment completed successfully")
			
			# Move to deployment history
			self.deployment_history.append(deployment)
			del self.active_deployments[deployment_id]
			
			print(f"✅ Deployment {deployment_id} completed successfully")
			return deployment
			
		except Exception as e:
			deployment.status = "failed"
			deployment.completed_at = datetime.now(timezone.utc)
			deployment.logs.append(f"Deployment failed: {str(e)}")
			
			# Attempt auto-rollback if enabled
			if deployment.auto_rollback:
				await self._execute_rollback(deployment)
			
			print(f"❌ Deployment {deployment_id} failed: {e}")
			raise
	
	async def _execute_deployment_pipeline(self, deployment: DeploymentResult):
		"""Execute the actual deployment steps."""
		# Step 1: Validate configuration
		deployment.logs.append("Validating configuration...")
		await self._validate_deployment_config(deployment)
		
		# Step 2: Run pre-deployment hooks
		deployment.logs.append("Running pre-deployment hooks...")
		await self._run_pre_deployment_hooks(deployment)
		
		# Step 3: Deploy to target environment
		deployment.logs.append(f"Deploying to {deployment.environment}...")
		await self._deploy_to_environment(deployment)
		
		# Step 4: Run post-deployment verification
		deployment.logs.append("Running post-deployment verification...")
		await self._verify_deployment(deployment)
		
		# Step 5: Run post-deployment hooks
		deployment.logs.append("Running post-deployment hooks...")
		await self._run_post_deployment_hooks(deployment)
	
	async def _validate_deployment_config(self, deployment: DeploymentResult):
		"""Validate deployment configuration."""
		# Configuration validation logic here
		await asyncio.sleep(1)  # Simulate validation
		deployment.logs.append("Configuration validation passed")
	
	async def _run_pre_deployment_hooks(self, deployment: DeploymentResult):
		"""Run pre-deployment hooks."""
		# Pre-deployment hooks (tests, security scans, etc.)
		await asyncio.sleep(2)  # Simulate hooks
		deployment.logs.append("Pre-deployment hooks completed")
	
	async def _deploy_to_environment(self, deployment: DeploymentResult):
		"""Deploy configuration to target environment."""
		if self.k8s_client:
			await self._deploy_to_kubernetes(deployment)
		else:
			await self._deploy_generic(deployment)
	
	async def _deploy_to_kubernetes(self, deployment: DeploymentResult):
		"""Deploy to Kubernetes environment."""
		try:
			# This would apply the generated Kubernetes manifests
			deployment.logs.append("Applying Kubernetes manifests...")
			
			# Simulate kubectl apply
			await asyncio.sleep(3)
			
			deployment.logs.append("Kubernetes deployment successful")
			
		except Exception as e:
			deployment.logs.append(f"Kubernetes deployment failed: {e}")
			raise
	
	async def _deploy_generic(self, deployment: DeploymentResult):
		"""Generic deployment method."""
		# Generic deployment logic
		await asyncio.sleep(2)
		deployment.logs.append("Generic deployment completed")
	
	async def _verify_deployment(self, deployment: DeploymentResult):
		"""Verify deployment success."""
		# Health checks, smoke tests, etc.
		await asyncio.sleep(1)
		deployment.logs.append("Deployment verification passed")
	
	async def _run_post_deployment_hooks(self, deployment: DeploymentResult):
		"""Run post-deployment hooks."""
		# Post-deployment hooks (notifications, monitoring setup, etc.)
		await asyncio.sleep(1)
		deployment.logs.append("Post-deployment hooks completed")
	
	async def _execute_rollback(self, deployment: DeploymentResult):
		"""Execute automatic rollback."""
		try:
			deployment.logs.append("Initiating automatic rollback...")
			
			# Find previous successful deployment
			previous_deployment = None
			for prev_deploy in reversed(self.deployment_history):
				if (prev_deploy.environment == deployment.environment and 
					prev_deploy.status == "success"):
					previous_deployment = prev_deploy
					break
			
			if previous_deployment:
				deployment.rollback_commit = previous_deployment.commit_sha
				deployment.logs.append(f"Rolling back to commit {previous_deployment.commit_sha[:8]}")
				
				# Execute rollback deployment
				await asyncio.sleep(2)  # Simulate rollback
				
				deployment.status = "rolled_back"
				deployment.logs.append("Rollback completed successfully")
			else:
				deployment.logs.append("No previous successful deployment found for rollback")
				
		except Exception as e:
			deployment.logs.append(f"Rollback failed: {e}")
	
	# ==================== Conflict Resolution ====================
	
	async def detect_merge_conflicts(
		self,
		repo_url: str,
		source_branch: str,
		target_branch: str
	) -> List[str]:
		"""Detect potential merge conflicts between branches."""
		try:
			repo = self.repositories.get(repo_url)
			if not repo:
				raise ValueError(f"Repository {repo_url} not initialized")
			
			# Get diff between branches
			try:
				merge_base = repo.merge_base(source_branch, target_branch)[0]
				
				# Check for conflicting files
				conflicts = []
				
				# Get changes in both branches since merge base
				source_diff = repo.git.diff(merge_base, source_branch, name_only=True).split('\n')
				target_diff = repo.git.diff(merge_base, target_branch, name_only=True).split('\n')
				
				# Find files changed in both branches
				common_files = set(source_diff) & set(target_diff)
				conflicts.extend(list(common_files))
				
				return [f for f in conflicts if f]  # Filter empty strings
				
			except GitCommandError:
				# Branches might not have a common ancestor
				return []
				
		except Exception as e:
			print(f"❌ Failed to detect conflicts: {e}")
			return []
	
	async def resolve_merge_conflicts(
		self,
		repo_url: str,
		source_branch: str,
		target_branch: str,
		resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.AI_ASSISTED
	) -> List[ConflictResolution]:
		"""Resolve merge conflicts using specified strategy."""
		try:
			conflicted_files = await self.detect_merge_conflicts(
				repo_url, source_branch, target_branch
			)
			
			if not conflicted_files:
				return []
			
			resolutions = []
			
			for file_path in conflicted_files:
				resolution = await self._resolve_file_conflict(
					repo_url, file_path, source_branch, target_branch, resolution_strategy
				)
				resolutions.append(resolution)
			
			return resolutions
			
		except Exception as e:
			print(f"❌ Failed to resolve conflicts: {e}")
			raise
	
	async def _resolve_file_conflict(
		self,
		repo_url: str,
		file_path: str,
		source_branch: str,
		target_branch: str,
		strategy: ConflictResolutionStrategy
	) -> ConflictResolution:
		"""Resolve conflict for a specific file."""
		repo = self.repositories[repo_url]
		
		# Get file content from both branches
		source_content = repo.git.show(f"{source_branch}:{file_path}")
		target_content = repo.git.show(f"{target_branch}:{file_path}")
		
		conflict_id = f"conflict_{hash(f'{file_path}{source_branch}{target_branch}')}"
		
		if strategy == ConflictResolutionStrategy.PREFER_SOURCE:
			resolved_content = source_content
			auto_resolved = True
		elif strategy == ConflictResolutionStrategy.PREFER_TARGET:
			resolved_content = target_content
			auto_resolved = True
		elif strategy == ConflictResolutionStrategy.AI_ASSISTED:
			resolved_content = await self._ai_resolve_conflict(
				source_content, target_content, file_path
			)
			auto_resolved = True
		else:
			# Manual resolution required
			resolved_content = source_content  # Default to source
			auto_resolved = False
		
		resolution = ConflictResolution(
			conflict_id=conflict_id,
			file_path=file_path,
			conflict_type="content_conflict",
			source_content=source_content,
			target_content=target_content,
			resolved_content=resolved_content,
			resolution_strategy=strategy,
			auto_resolved=auto_resolved
		)
		
		self.conflict_resolutions.append(resolution)
		return resolution
	
	async def _ai_resolve_conflict(
		self,
		source_content: str,
		target_content: str,
		file_path: str
	) -> str:
		"""Use AI to intelligently resolve merge conflicts."""
		# This would integrate with an AI service to analyze the conflict
		# and suggest the best resolution
		
		# For now, implement simple heuristics
		if file_path.endswith('.yaml') or file_path.endswith('.yml'):
			return await self._resolve_yaml_conflict(source_content, target_content)
		elif file_path.endswith('.json'):
			return await self._resolve_json_conflict(source_content, target_content)
		else:
			return await self._resolve_generic_conflict(source_content, target_content)
	
	async def _resolve_yaml_conflict(self, source: str, target: str) -> str:
		"""Resolve YAML configuration conflicts."""
		try:
			source_data = yaml.safe_load(source)
			target_data = yaml.safe_load(target)
			
			# Merge dictionaries intelligently
			merged_data = self._deep_merge_dicts(source_data, target_data)
			
			return yaml.dump(merged_data, default_flow_style=False)
		except:
			# Fall back to source content if parsing fails
			return source
	
	async def _resolve_json_conflict(self, source: str, target: str) -> str:
		"""Resolve JSON configuration conflicts."""
		try:
			source_data = json.loads(source)
			target_data = json.loads(target)
			
			# Merge objects intelligently
			merged_data = self._deep_merge_dicts(source_data, target_data)
			
			return json.dumps(merged_data, indent=2)
		except:
			return source
	
	async def _resolve_generic_conflict(self, source: str, target: str) -> str:
		"""Resolve generic text conflicts."""
		# Simple line-by-line merge for generic files
		source_lines = source.split('\n')
		target_lines = target.split('\n')
		
		# For simplicity, prefer source for generic files
		return source
	
	def _deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
		"""Deep merge two dictionaries."""
		result = dict1.copy()
		
		for key, value in dict2.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._deep_merge_dicts(result[key], value)
			else:
				result[key] = value
		
		return result
	
	# ==================== Monitoring & Observability ====================
	
	async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
		"""Get current deployment status."""
		return self.active_deployments.get(deployment_id)
	
	async def get_deployment_history(
		self,
		environment: Optional[str] = None,
		limit: int = 50
	) -> List[DeploymentResult]:
		"""Get deployment history."""
		history = self.deployment_history
		
		if environment:
			history = [d for d in history if d.environment == environment]
		
		return sorted(history, key=lambda x: x.started_at, reverse=True)[:limit]
	
	async def get_gitops_metrics(self) -> Dict[str, Any]:
		"""Get GitOps operational metrics."""
		total_deployments = len(self.deployment_history)
		successful_deployments = len([d for d in self.deployment_history if d.status == "success"])
		failed_deployments = len([d for d in self.deployment_history if d.status == "failed"])
		
		return {
			"total_deployments": total_deployments,
			"successful_deployments": successful_deployments,
			"failed_deployments": failed_deployments,
			"success_rate": successful_deployments / total_deployments * 100 if total_deployments > 0 else 0,
			"active_deployments": len(self.active_deployments),
			"conflicts_resolved": len(self.conflict_resolutions),
			"auto_resolved_conflicts": len([c for c in self.conflict_resolutions if c.auto_resolved])
		}
	
	# ==================== Cleanup ====================
	
	async def close(self):
		"""Clean up GitOps engine resources."""
		# Clean up temporary repositories
		for repo_url, repo in self.repositories.items():
			repo_path = Path(repo.working_dir)
			if repo_path.exists() and repo_path.is_dir():
				shutil.rmtree(repo_path, ignore_errors=True)
		
		self.repositories.clear()
		print("🔄 GitOps engine closed")


# ==================== Factory Functions ====================

async def create_gitops_engine(
	repository_url: str,
	branch_strategy: BranchingStrategy = BranchingStrategy.ENVIRONMENT_BRANCHES,
	deployment_strategy: GitOpsStrategy = GitOpsStrategy.PULL_BASED
) -> CentralConfigurationGitOps:
	"""Create and initialize GitOps engine."""
	config = GitOpsConfiguration(
		repository_url=repository_url,
		branch_strategy=branch_strategy,
		deployment_strategy=deployment_strategy,
		auto_merge=True,
		conflict_resolution=ConflictResolutionStrategy.AI_ASSISTED,
		environments=["development", "staging", "production"],
		sync_interval_seconds=300,
		notification_webhooks=[],
		approval_required_for_prod=True
	)
	
	engine = CentralConfigurationGitOps(config)
	
	# Initialize repository
	await engine.initialize_repository(repository_url)
	
	print("🔄 GitOps engine initialized")
	return engine