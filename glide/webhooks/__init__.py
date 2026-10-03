"""Optional webhook ingress, durable queue and headless worker.

Off by default (D5): nothing here starts a server, a worker or a desktop run on import, and the service refuses
to build until the configuration says `"enabled": true`. See `glide/webhooks/README.md`.
"""
