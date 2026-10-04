# Nexus AI

A multi-service AI platform with an MCP server, backend proxy, model orchestration, shared database and cache modules, and a React administration portal.

## Usage

Review the Docker Compose configuration and supply service credentials outside version control. The administration app runs from `admin_portal` with `npm ci` and `npm run dev`; validate it with `npm run build`. Python service dependencies are listed in `backend_proxy/requirements.txt` and `mcp_server/requirements.txt`.


## Author

[rajivranjanmars](https://rajivranjana.in)
