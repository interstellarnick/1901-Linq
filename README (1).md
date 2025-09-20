
# Admin Contacts — Interactive Report

An interactive Streamlit app with filtering, search, and a pie chart for **Created By User**.

## Quick start (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Free hosting options

### Option A — Streamlit Community Cloud (easiest)
1. Create a **new GitHub repo** and add all files in this folder.
2. Go to https://share.streamlit.io (Streamlit Community Cloud) and **Deploy an app**.
3. Pick your repo, branch, and `app.py` as the entrypoint.
4. Click **Deploy** — you’ll get a shareable URL.

### Option B — Hugging Face Spaces (also free)
1. Create a new **Space** → Template: **Streamlit**.
2. Upload these files (or link your GitHub repo).
3. Spaces auto-builds and hosts your app with a public URL.

### Option C — Static HTML fallback (no Python server)
If you need a single-file report, ask me and I’ll generate a `static_report.html` with basic filters and the pie chart that you can host on **GitHub Pages**. (It’s less flexible than the Streamlit app.)
