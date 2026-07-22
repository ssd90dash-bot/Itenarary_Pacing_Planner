"""HTML + Tailwind frontend for the itinerary planner.

A thin FastAPI adapter over the `planner/` package: parse the request, call the
same domain logic the Streamlit app uses, render a Jinja2 template. No planning
logic lives here.
"""
