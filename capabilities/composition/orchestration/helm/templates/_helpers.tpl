{{/*
Expand the name of the chart.
*/}}
{{- define "apg-workflow.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "apg-workflow.fullname" -}}
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
{{- define "apg-workflow.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "apg-workflow.labels" -}}
helm.sh/chart: {{ include "apg-workflow.chart" . }}
{{ include "apg-workflow.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: apg-platform
{{- end }}

{{/*
Selector labels
*/}}
{{- define "apg-workflow.selectorLabels" -}}
app.kubernetes.io/name: {{ include "apg-workflow.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "apg-workflow.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "apg-workflow.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL host
*/}}
{{- define "apg-workflow.postgresql.host" -}}
{{- if .Values.database.external.enabled }}
{{- .Values.database.external.host }}
{{- else }}
{{- printf "%s-postgresql" (include "apg-workflow.fullname" .) }}
{{- end }}
{{- end }}

{{/*
PostgreSQL secret name
*/}}
{{- define "apg-workflow.postgresql.secretName" -}}
{{- if .Values.database.external.enabled }}
{{- printf "%s-external-db" (include "apg-workflow.fullname" .) }}
{{- else }}
{{- printf "%s-postgresql" (include "apg-workflow.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Redis host
*/}}
{{- define "apg-workflow.redis.host" -}}
{{- if .Values.redis.external.enabled }}
{{- .Values.redis.external.host }}
{{- else }}
{{- printf "%s-redis-master" (include "apg-workflow.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Redis secret name
*/}}
{{- define "apg-workflow.redis.secretName" -}}
{{- if .Values.redis.external.enabled }}
{{- printf "%s-external-redis" (include "apg-workflow.fullname" .) }}
{{- else }}
{{- printf "%s-redis" (include "apg-workflow.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Common annotations
*/}}
{{- define "apg-workflow.annotations" -}}
meta.helm.sh/release-name: {{ .Release.Name }}
meta.helm.sh/release-namespace: {{ .Release.Namespace }}
{{- if .Values.annotations }}
{{ toYaml .Values.annotations }}
{{- end }}
{{- end }}

{{/*
Generate certificates for webhook
*/}}
{{- define "apg-workflow.gen-certs" -}}
{{- $altNames := list ( printf "%s.%s" (include "apg-workflow.name" .) .Release.Namespace ) ( printf "%s.%s.svc" (include "apg-workflow.name" .) .Release.Namespace ) -}}
{{- $ca := genCA "apg-workflow-ca" 3650 -}}
{{- $cert := genSignedCert ( include "apg-workflow.name" . ) nil $altNames 3650 $ca -}}
tls.crt: {{ $cert.Cert | b64enc }}
tls.key: {{ $cert.Key | b64enc }}
ca.crt: {{ $ca.Cert | b64enc }}
{{- end }}

{{/*
Create environment variables for database connection
*/}}
{{- define "apg-workflow.database.env" -}}
{{- if .Values.database.enabled }}
- name: DATABASE_HOST
  value: {{ include "apg-workflow.postgresql.host" . | quote }}
- name: DATABASE_PORT
  value: "5432"
{{- if .Values.database.external.enabled }}
- name: DATABASE_NAME
  value: {{ .Values.database.external.database | quote }}
- name: DATABASE_USER
  value: {{ .Values.database.external.username | quote }}
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "apg-workflow.postgresql.secretName" . }}
      key: password
{{- else }}
- name: DATABASE_NAME
  value: {{ .Values.database.postgresql.auth.database | quote }}
- name: DATABASE_USER
  value: {{ .Values.database.postgresql.auth.username | quote }}
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "apg-workflow.postgresql.secretName" . }}
      key: postgres-password
{{- end }}
{{- end }}
{{- end }}

{{/*
Create environment variables for Redis connection
*/}}
{{- define "apg-workflow.redis.env" -}}
{{- if .Values.redis.enabled }}
- name: REDIS_HOST
  value: {{ include "apg-workflow.redis.host" . | quote }}
- name: REDIS_PORT
  value: "6379"
{{- if .Values.redis.auth.enabled }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "apg-workflow.redis.secretName" . }}
      key: redis-password
{{- end }}
{{- end }}
{{- end }}

{{/*
Create image pull policy
*/}}
{{- define "apg-workflow.imagePullPolicy" -}}
{{- if .Values.development.enabled }}
imagePullPolicy: Always
{{- else }}
imagePullPolicy: {{ .Values.app.image.pullPolicy }}
{{- end }}
{{- end }}

{{/*
Create resource requirements
*/}}
{{- define "apg-workflow.resources" -}}
{{- if .Values.development.enabled }}
resources:
  {{- toYaml .Values.development.resources | nindent 2 }}
{{- else }}
resources:
  {{- toYaml .Values.app.resources | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Create ingress hostname
*/}}
{{- define "apg-workflow.ingress.hostname" -}}
{{- if .Values.development.enabled }}
{{- .Values.development.ingress.host }}
{{- else }}
{{- .Values.ingress.main.host }}
{{- end }}
{{- end }}

{{/*
Create probe configuration
*/}}
{{- define "apg-workflow.probes" -}}
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  successThreshold: 1
  failureThreshold: 3
readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  successThreshold: 1
  failureThreshold: 3
startupProbe:
  httpGet:
    path: /startup
    port: http
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  successThreshold: 1
  failureThreshold: 30
{{- end }}

{{/*
Create volume mounts
*/}}
{{- define "apg-workflow.volumeMounts" -}}
{{- if .Values.storage.data.enabled }}
- name: data-storage
  mountPath: /app/data
{{- end }}
{{- if .Values.storage.models.enabled }}
- name: models-storage
  mountPath: /app/models
{{- end }}
{{- if .Values.storage.templates.enabled }}
- name: templates-storage
  mountPath: /app/templates
{{- end }}
- name: temp-storage
  mountPath: /tmp
- name: logs-storage
  mountPath: /app/logs
{{- end }}

{{/*
Create volumes
*/}}
{{- define "apg-workflow.volumes" -}}
{{- if .Values.storage.data.enabled }}
- name: data-storage
  persistentVolumeClaim:
    claimName: {{ include "apg-workflow.fullname" . }}-data-pvc
{{- end }}
{{- if .Values.storage.models.enabled }}
- name: models-storage
  persistentVolumeClaim:
    claimName: {{ include "apg-workflow.fullname" . }}-models-pvc
{{- end }}
{{- if .Values.storage.templates.enabled }}
- name: templates-storage
  persistentVolumeClaim:
    claimName: {{ include "apg-workflow.fullname" . }}-templates-pvc
{{- end }}
- name: temp-storage
  emptyDir:
    sizeLimit: 1Gi
- name: logs-storage
  emptyDir: {}
{{- end }}

{{/*
Create security context
*/}}
{{- define "apg-workflow.securityContext" -}}
securityContext:
  {{- toYaml .Values.security.securityContext | nindent 2 }}
{{- end }}

{{/*
Create pod security context
*/}}
{{- define "apg-workflow.podSecurityContext" -}}
securityContext:
  {{- toYaml .Values.security.podSecurityContext | nindent 2 }}
{{- end }}

{{/*
Create affinity rules
*/}}
{{- define "apg-workflow.affinity" -}}
{{- if .Values.ha.podAntiAffinity.enabled }}
affinity:
  podAntiAffinity:
    {{- if eq .Values.ha.podAntiAffinity.type "required" }}
    requiredDuringSchedulingIgnoredDuringExecution:
    {{- else }}
    preferredDuringSchedulingIgnoredDuringExecution:
    {{- end }}
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - {{ include "apg-workflow.name" . }}
        topologyKey: kubernetes.io/hostname
{{- end }}
{{- with .Values.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Validate configuration
*/}}
{{- define "apg-workflow.validate" -}}
{{- if and .Values.database.external.enabled (not .Values.database.external.host) }}
{{- fail "External database host must be specified when external database is enabled" }}
{{- end }}
{{- if and .Values.redis.external.enabled (not .Values.redis.external.host) }}
{{- fail "External Redis host must be specified when external Redis is enabled" }}
{{- end }}
{{- if and .Values.ingress.enabled (not .Values.ingress.main.host) }}
{{- fail "Ingress host must be specified when ingress is enabled" }}
{{- end }}
{{- end }}