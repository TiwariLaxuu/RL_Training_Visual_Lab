"""Launch the RL Visual Training Lab.

    python run_dashboard.py

Then open http://127.0.0.1:8000 in a browser.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("rl_dashboard.server:app", host="127.0.0.1", port=8001, reload=False)
