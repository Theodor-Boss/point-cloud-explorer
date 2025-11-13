# 3D Point Cloud Viewer

A web-based 3D viewer for visualizing point clouds from Python, with first-person controls.

## Complete Beginner Setup (Linux)

### Step 1: Install Node.js and npm

Node.js is required to run the web viewer. `npm` is the package manager that comes with it.

```bash
# Check if you already have Node.js installed
node --version
npm --version

# If not installed, install using your package manager:
# Ubuntu/Debian:
sudo apt update
sudo apt install nodejs npm

# Fedora:
sudo dnf install nodejs npm

# Arch:
sudo pacman -S nodejs npm
```

### Step 2: Get the Code

**Option A: Clone from GitHub (if connected)**

```bash
git clone <YOUR_GIT_URL>
cd <YOUR_PROJECT_NAME>
```

**Option B: Download from Lovable**

1. Go to https://lovable.dev/projects/7afcf1b3-d095-469b-a275-7b8e938741df
2. Click the download/export button to get a ZIP file
3. Extract the ZIP file
4. Open terminal in that folder

### Step 3: Install Dependencies

```bash
# This downloads all the JavaScript libraries needed
npm install
```

This will take a few minutes the first time.

### Step 4: Start the Web Viewer

```bash
# This starts a local web server
npm run dev
```

You should see output like:
```
  ➜  Local:   http://localhost:8080/
```

**Keep this terminal window open!** The web server runs here.

### Step 5: Open in Browser

Open Firefox (or any browser) and go to:
```
http://localhost:8080
```

You should see the 3D viewer!

### Step 6: Run Your Python Code

In a **new terminal window** (keep the web server running):

```bash
# Make sure you have the Python script
ls python_export.py  # Should exist

# Run your Python code
python3 your_script.py
```

This will create `pointcloud.json` and `lines.json` files.

### Step 7: Load Your Data

In the browser at `http://localhost:8080`:
1. You'll see a "Load Point Cloud" card
2. Click "Choose File" or drag & drop `pointcloud.json`
3. Your points will appear in 3D!

### Controls

- **Click anywhere** - Lock mouse to look around
- **WASD** - Move forward/left/back/right  
- **Space** - Move up
- **Shift** - Move down
- **ESC** - Unlock mouse

## Project info

**URL**: https://lovable.dev/projects/7afcf1b3-d095-469b-a275-7b8e938741df

## For Python Integration

See [PYTHON_INTEGRATION.md](PYTHON_INTEGRATION.md) for details on exporting point clouds from Python.

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/7afcf1b3-d095-469b-a275-7b8e938741df) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/7afcf1b3-d095-469b-a275-7b8e938741df) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
