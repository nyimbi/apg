{{/*
APG Platform Helm Chart Helpers
*/}}

{{- define "apg-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "apg-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "apg-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "apg-platform.labels" -}}
helm.sh/chart: {{ include "apg-platform.chart" . }}
{{ include "apg-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: apg-platform
{{- end }}

{{- define "apg-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "apg-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Common environment variables for all APG services */}}
{{- define "apg-platform.commonEnv" -}}
- name: APG_DATABASE_URL
  value: "postgresql+asyncpg://{{ .Values.postgresql.auth.username }}:$(POSTGRES_PASSWORD)@{{ include "apg-platform.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}"
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "apg-platform.fullname" . }}-postgresql
      key: password
- name: NATS_URL
  value: "nats://{{ include "apg-platform.fullname" . }}-nats:4222"
- name: TEMPORAL_HOST
  value: "{{ include "apg-platform.fullname" . }}-temporal:7233"
- name: OPA_URL
  value: "http://{{ include "apg-platform.fullname" . }}-opa:8181"
- name: APG_DEFAULT_TENANT
  value: {{ .Values.global.defaultTenant | quote }}
{{- if .Values.ollama.enabled }}
- name: OLLAMA_BASE_URL
  value: "http://{{ include "apg-platform.fullname" . }}-ollama:11434"
{{- end }}
{{- end }}
