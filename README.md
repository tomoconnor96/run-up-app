# Run Up — bowling action analysis (v0)

A working web app: upload a delivery clip, get it broken into phases
(run-up, gather, back foot contact, front foot contact, release, follow-through),
scored, with an annotated slow-motion video and a written report.

This has been tested locally end-to-end. What's left is putting it on a server
that's on all the time, so you get a real link to share. That takes about
15 minutes and doesn't need any coding — just clicking through two free
websites. Steps below.

## What you need
- A Google/email address for signing up (free)
- These files, which you already have

## Step 1 — Put the code on GitHub (free)
GitHub is just a place to store the code so the hosting service can find it.

1. Go to **github.com** and create a free account if you don't have one.
2. Click the **+** in the top right → **New repository**.
3. Name it something like `run-up-app`, keep it **Public**, click **Create repository**.
4. On the new repo's page, click **uploading an existing file**.
5. Drag this whole folder (all the files and the `engine`, `static`, `templates` sub-folders)
   into the upload box. GitHub will keep the folder structure.
6. Scroll down, click **Commit changes**.

## Step 2 — Deploy it on Render (free tier)
Render is the hosting service that keeps the app running and gives you a public link.

1. Go to **render.com** and sign up free (you can sign up directly with your GitHub account,
   which also connects the two automatically).
2. Click **New +** → **Web Service**.
3. Choose the `run-up-app` repository you just created.
4. Render will detect the `Dockerfile` automatically — leave the settings as they are.
5. Choose the **Free** instance type.
6. Click **Create Web Service**.
7. Wait 3–5 minutes while it builds. When it's done, Render shows you a link like
   `https://run-up-app.onrender.com` — that's your live app. Share that link with anyone.

**Good to know about the free tier:** it goes to sleep after 15 minutes of no traffic,
so the first visit after a break takes ~30 seconds to wake up — that's normal, not a bug.
If that's annoying, Render's paid tier (~$7/month) keeps it always-on.

## Trying it yourself first (optional, needs a developer/terminal)
```
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000` in a browser.

## Current limits, honestly
- Works best with the bowler filmed **side-on**, full run-up to follow-through in frame,
  without other people moving close to the camera.
- Clips over 12 seconds aren't accepted yet (keep it to the run-up through follow-through).
- Scoring is from motion-tracking, not full joint/skeleton tracking yet — see the
  in-app caveat on the results page. This is the natural next upgrade.
- Every upload is processed fresh; nothing is saved permanently on the free tier
  (files can be cleared when the service restarts).
