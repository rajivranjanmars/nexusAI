FROM node:26-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Run Vite dev server (accessible from host)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
