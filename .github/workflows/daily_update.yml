name: Daily Vegetable Price Update (Open Source AI)

on:
  schedule:
    # 06:00 UTC = 11:30 AM IST
    - cron: "0 6 * * *"
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  update-prices:
    runs-on: ubuntu-latest
    timeout-minutes: 45   # Ollama + model pull needs extra time

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install Python dependencies
        run: pip install -r requirements.txt

      # ── Ollama cache: saves ~4GB re-download each run ──────────────────
      - name: Cache Ollama model
        uses: actions/cache@v4
        id: ollama-cache
        with:
          path: ~/.ollama
          key: ollama-${{ env.OLLAMA_MODEL }}-v1
          restore-keys: |
            ollama-${{ env.OLLAMA_MODEL }}-
        env:
          OLLAMA_MODEL: ${{ vars.OLLAMA_MODEL || 'mistral' }}

      # ── Install & start Ollama as a background service ──────────────────
      - name: Install Ollama
        run: curl -fsSL https://ollama.com/install.sh | sh

      - name: Start Ollama server (background)
        run: |
          ollama serve &
          echo "Ollama PID: $!"
          # Give it a moment before pulling
          sleep 5

      # ── Transcript scraper ──────────────────────────────────────────────
      - name: Restore transcript CSV from cache
        uses: actions/cache@v4
        with:
          path: data/market_transcripts_master.csv
          key: transcript-csv-${{ runner.os }}-${{ github.run_id }}
          restore-keys: |
            transcript-csv-${{ runner.os }}-

      - name: Run transcript scraper
        env:
          CHANNEL_ID: ${{ secrets.CHANNEL_ID || 'UCxEW_BSHnu43J8-ANnSJ80w' }}
          MAX_NEW_VIDEOS: "5"
        run: |
          mkdir -p data
          python scraper.py

      # ── Price extraction with open-source model ─────────────────────────
      - name: Extract vegetable prices (Mistral via Ollama)
        env:
          # Change to "gemma2:2b" for faster/lighter or "llama3.1" for best accuracy
          OLLAMA_MODEL: ${{ vars.OLLAMA_MODEL || 'mistral' }}
        run: python parse_prices.py

      # ── Commit & push data back to repo ────────────────────────────────
      - name: Commit updated data files
        run: |
          git config --global user.name  "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add data/market_transcripts_master.csv data/prices.json || true
          git diff --staged --quiet || git commit -m "chore: update mandi prices [$(date -u +'%Y-%m-%d')] via ${OLLAMA_MODEL:-mistral}"
          git push
        env:
          OLLAMA_MODEL: ${{ vars.OLLAMA_MODEL || 'mistral' }}

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: dashboard/

  deploy-pages:
    needs: update-prices
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
