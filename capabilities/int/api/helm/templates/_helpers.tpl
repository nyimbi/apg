{{/*
Expand the name of the chart.
*/}}
{{- define "integration-api-management.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "integration-api-management.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "integration-api-management.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "integration-api-management.labels" -}}
helm.sh/chart: {{ include "integration-api-management.chart" . }}
{{ include "integration-api-management.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
apg.datacraft.co.ke/capability-id: integration-api-management
apg.datacraft.co.ke/capability-version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "integration-api-management.selectorLabels" -}}
app.kubernetes.io/name: {{ include "integration-api-management.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "integration-api-management.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "integration-api-management.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL fullname
*/}}
{{- define "integration-api-management.postgresql.fullname" -}}
{{- printf "%s-postgresql" (include "integration-api-management.fullname" .) }}
{{- end }}

{{/*
Redis fullname
*/}}
{{- define "integration-api-management.redis.fullname" -}}
{{- printf "%s-redis" (include "integration-api-management.fullname" .) }}
{{- end }}

{{/*
Database URL for external or internal PostgreSQL
*/}}
{{- define "integration-api-management.database.url" -}}
{{- if .Values.database.external.enabled }}
{{- printf "postgresql://%s:%s@%s:%d/%s?sslmode=%s" .Values.database.external.username .Values.database.external.password .Values.database.external.host .Values.database.external.port .Values.database.external.database .Values.database.external.sslMode }}
{{- else }}
{{- printf "postgresql://%s:%s@%s:5432/%s?sslmode=require" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "integration-api-management.postgresql.fullname" .) .Values.postgresql.auth.database }}
{{- end }}
{{- end }}

{{/*
Redis URL for external or internal Redis
*/}}
{{- define "integration-api-management.redis.url" -}}
{{- if .Values.cache.external.enabled }}
{{- if .Values.cache.external.password }}
{{- printf "redis://:%s@%s:%d/%d" .Values.cache.external.password .Values.cache.external.host .Values.cache.external.port .Values.cache.external.database }}
{{- else }}
{{- printf "redis://%s:%d/%d" .Values.cache.external.host .Values.cache.external.port .Values.cache.external.database }}
{{- end }}
{{- else }}
{{- printf "redis://:%s@%s:6379/0" .Values.redis.auth.password (include "integration-api-management.redis.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Generate certificates
*/}}
{{- define "integration-api-management.gen-certs" -}}
{{- $ca := genCA "integration-api-management-ca" 3650 }}
{{- $cert := genSignedCert (include "integration-api-management.fullname" .) nil (list (printf "%s.%s" (include "integration-api-management.fullname" .) .Release.Namespace) (printf "%s.%s.svc" (include "integration-api-management.fullname" .) .Release.Namespace)) 3650 $ca }}
tls.crt: {{ $cert.Cert | b64enc }}
tls.key: {{ $cert.Key | b64enc }}
ca.crt: {{ $ca.Cert | b64enc }}
{{- end }}

{{/*
Validation functions
*/}}
{{- define "integration-api-management.validate.replicas" -}}
{{- if lt (.Values.app.replicaCount | int) 1 }}
{{- fail "app.replicaCount must be at least 1" }}
{{- end }}
{{- if and .Values.app.autoscaling.enabled (lt (.Values.app.autoscaling.minReplicas | int) 1) }}
{{- fail "app.autoscaling.minReplicas must be at least 1" }}
{{- end }}
{{- if and .Values.app.autoscaling.enabled (gt (.Values.app.autoscaling.minReplicas | int) (.Values.app.autoscaling.maxReplicas | int)) }}
{{- fail "app.autoscaling.minReplicas must be less than or equal to maxReplicas" }}
{{- end }}
{{- end }}

{{/*
Resource validation
*/}}
{{- define "integration-api-management.validate.resources" -}}
{{- if .Values.app.resources.limits }}
{{- if .Values.app.resources.requests }}
{{- $limitsCpu := .Values.app.resources.limits.cpu | toString }}
{{- $requestsCpu := .Values.app.resources.requests.cpu | toString }}
{{- $limitsMemory := .Values.app.resources.limits.memory | toString }}
{{- $requestsMemory := .Values.app.resources.requests.memory | toString }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "integration-api-management.commonEnv" -}}
- name: ENVIRONMENT
  value: {{ .Values.app.environment }}
- name: DEBUG
  value: "{{ .Values.app.debug }}"
- name: NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
{{- end }}

{{/*
APG Platform environment variables
*/}}
{{- define "integration-api-management.apgEnv" -}}
{{- if .Values.global.apgPlatform.enabled }}
- name: APG_PLATFORM_ENABLED
  value: "true"
- name: APG_PLATFORM_NAMESPACE
  value: {{ .Values.global.apgPlatform.namespace }}
{{- if .Values.global.apgPlatform.serviceDiscovery.enabled }}
- name: SERVICE_DISCOVERY_ENABLED
  value: "true"
- name: CONSUL_ADDRESS
  value: {{ .Values.global.apgPlatform.serviceDiscovery.consul.address }}
{{- end }}
{{- if .Values.global.apgPlatform.eventBus.enabled }}
- name: EVENT_BUS_ENABLED
  value: "true"
- name: KAFKA_BROKERS
  value: {{ .Values.global.apgPlatform.eventBus.kafka.brokers }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Security context validation
*/}}
{{- define "integration-api-management.validate.security" -}}
{{- if not .Values.app.securityContext.runAsNonRoot }}
{{- fail "securityContext.runAsNonRoot must be true for security compliance" }}
{{- end }}
{{- if eq (.Values.app.securityContext.runAsUser | int) 0 }}
{{- fail "securityContext.runAsUser cannot be 0 (root) for security compliance" }}
{{- end }}
{{- end }}

{{/*
Network policy rules
*/}}
{{- define "integration-api-management.networkPolicy.ingress" -}}
- from:
  - namespaceSelector:
      matchLabels:
        name: kube-system
  - namespaceSelector:
      matchLabels:
        name: {{ .Values.global.apgPlatform.namespace | default "apg-platform" }}
  - podSelector:
      matchLabels:
        component: load-balancer
  ports:
  - protocol: TCP
    port: {{ .Values.service.apiManagement.targetPort }}
  - protocol: TCP
    port: {{ .Values.service.gateway.targetPort }}
  - protocol: TCP
    port: {{ .Values.service.metrics.targetPort }}
{{- end }}

{{/*
Network policy egress rules
*/}}
{{- define "integration-api-management.networkPolicy.egress" -}}
- to:
  - podSelector:
      matchLabels:
        app.kubernetes.io/name: postgresql
  ports:
  - protocol: TCP
    port: 5432
- to:
  - podSelector:
      matchLabels:
        app.kubernetes.io/name: redis
  ports:
  - protocol: TCP
    port: 6379
- {}  # Allow all egress for external API calls
{{- end }}