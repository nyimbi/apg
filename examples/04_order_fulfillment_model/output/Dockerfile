FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APG_HOST=0.0.0.0
ENV APG_PORT=8080

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python app.py --self-test >/tmp/order_fulfillment_self_test.json || exit 1

CMD ["python", "app.py"]
