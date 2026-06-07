FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APG_HOST=0.0.0.0
ENV APG_PORT=8080

WORKDIR /app
COPY . /app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python app.py --self-test >/tmp/localised_inventory_self_test.json || exit 1

CMD ["python", "app.py"]
