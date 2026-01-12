# Fake Job Detection - Frontend

A React-based frontend application for analyzing job postings to detect fraudulent listings using AI.

## Development

### Prerequisites
- Node.js 18+ and npm

### Setup
1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the frontend directory:
```env
VITE_API_URL=http://127.0.0.1:8000
```

3. Start the development server:
```bash
npm run dev
```

## Deployment to Vercel

### Option 1: Deploy via Vercel CLI
1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Navigate to the frontend directory:
```bash
cd frontend
```

3. Deploy:
```bash
vercel
```

### Option 2: Deploy via Vercel Dashboard
1. Push your code to GitHub/GitLab/Bitbucket
2. Go to [vercel.com](https://vercel.com) and import your repository
3. Configure the project:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

4. Add Environment Variable:
   - Go to Project Settings → Environment Variables
   - Add `VITE_API_URL` with your backend API URL (e.g., `https://your-api.vercel.app` or your backend server URL)

5. Deploy!

### Important Notes
- Make sure to set the `VITE_API_URL` environment variable in Vercel to point to your backend API
- The frontend uses Vite, which requires environment variables to be prefixed with `VITE_` to be accessible in the browser
- If your backend is on a different domain, ensure CORS is properly configured on your backend

## Build

To build for production:
```bash
npm run build
```

The output will be in the `dist` directory.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://127.0.0.1:8000` |

