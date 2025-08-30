## Rspeedy project

This is a ReactLynx project bootstrapped with `create-rspeedy`.

## Getting Started

First, install the dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Scan the QRCode in the terminal with your LynxExplorer App to see the result.

## Backend server

To make available the local AI model, we run a Flask server with POST requests exposed.

To run this server,

```
cd pythonFlaskServer
python3 FlaskTechJam2025.py
```

### Setting up the Flask IP address

It is very important to set the IP address to the one of your flask server.

This must be done for the SERVER_URL variable in `Gallery.tsx` and `UploadIcon.tsx`