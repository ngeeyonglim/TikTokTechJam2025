## Rspeedy project

This is a ReactLynx project bootstrapped with `create-rspeedy`.

## Getting Started

Ensure that you have the Lynx Explorer downloaded on your device. If you do not have the Lynx Explorer downloaded, you can refer to this guide to download it, [Lynx Explorer](https://lynxjs.org/guide/start/quick-start.html).

Once Lynx Explorer is installed on your device, proceed with the following instructions in your terminal.

First, install the dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Scan the QRCode in the terminal with your Lynx Explorer App to see the result.

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
