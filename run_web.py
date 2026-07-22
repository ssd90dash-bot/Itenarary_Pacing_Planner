"""Launch the HTML frontend.

    python run_web.py

Then open http://localhost:8000. The Streamlit app is unaffected and still runs
via `streamlit run app.py`.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.main:app", host="127.0.0.1", port=8000, reload=False)
