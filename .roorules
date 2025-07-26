  You are an expert in Python, FastAPI, and scalable API development.
  
  Key Principles
  - Write concise, technical responses with accurate Python examples.
  - Use functional, declarative programming; avoid classes where possible.
  - Prefer iteration and modularization over code duplication.
  - Use descriptive variable names with auxiliary verbs (e.g., is_active, has_permission).
  - Use lowercase with underscores for directories and files (e.g., routers/user_routes.py).
  - Favor named exports for routes and utility functions.
  - Use the Receive an Object, Return an Object (RORO) pattern.
  
  Python/FastAPI
  - Use def for pure functions and async def for asynchronous operations.
  - Use type hints for all function signatures. Prefer Pydantic models over raw dictionaries for input validation.
  - Use modern Pydantic v2 validators (`@field_validator`, `@model_validator`) instead of deprecated ones (`@validator`, `@root_validator`).
  - File structure: exported router, sub-routes, utilities, static content, types (models, schemas).
  - Avoid unnecessary curly braces in conditional statements.
  - For single-line statements in conditionals, omit curly braces.
  - Use concise, one-line syntax for simple conditional statements (e.g., if condition: do_something()).
  
  Error Handling and Validation
  - Prioritize error handling and edge cases:
    - Handle errors and edge cases at the beginning of functions.
    - Use early returns for error conditions to avoid deeply nested if statements.
    - Place the happy path last in the function for improved readability.
    - Avoid unnecessary else statements; use the if-return pattern instead.
    - Use guard clauses to handle preconditions and invalid states early.
    - Implement proper error logging and user-friendly error messages.
    - Use custom error types or error factories for consistent error handling.
  
  Dependencies
  - FastAPI
  - Pydantic v2
  - Async database libraries like asyncpg or aiomysql
  - SQLAlchemy 2.0 (if using ORM features)
  
  FastAPI-Specific Guidelines
  - Use functional components (plain functions) and Pydantic models for input validation and response schemas.
  - Use declarative route definitions with clear return type annotations.
  - Use def for synchronous operations and async def for asynchronous ones.
  - Minimize @app.on_event("startup") and @app.on_event("shutdown"); prefer lifespan context managers for managing startup and shutdown events.
  - Use middleware for logging, error monitoring, and performance optimization.
  - Optimize for performance using async functions for I/O-bound tasks, caching strategies, and lazy loading.
  - Use HTTPException for expected errors and model them as specific HTTP responses.
  - Use middleware for handling unexpected errors, logging, and error monitoring.
  - Use Pydantic's BaseModel for consistent input/output validation and response schemas.
  
  Pydantic v2 Validation
  - Use `@field_validator` for single-field validation and `@model_validator` for cross-field validation.
  - Specify the validation mode (`'before'`, `'after'`, `'wrap'`, `'plain'`) to control when the validator is run.
  - For reusable validators, define them as standalone functions and apply them using `Annotated`.
  - Use the `info` parameter in validators to access validation context.
  
  Performance Optimization
  - Minimize blocking I/O operations; use asynchronous operations for all database calls and external API requests.
  - Implement caching for static and frequently accessed data using tools like Redis or in-memory stores.
  - Optimize data serialization and deserialization with Pydantic.
  - Use lazy loading techniques for large datasets and substantial API responses.
  
  Key Conventions
  1. Rely on FastAPI’s dependency injection system for managing state and shared resources.
  2. Prioritize API performance metrics (response time, latency, throughput).
  3. Limit blocking operations in routes:
     - Favor asynchronous and non-blocking flows.
     - Use dedicated async functions for database and external API operations.
     - Structure routes and dependencies clearly to optimize readability and maintainability.
  
  Refer to FastAPI documentation for Data Models, Path Operations, and Middleware for best practices.

  Modern Python Stack Recommendations
  - HTTPX over Requests: For its async support, HTTP/2 capabilities, and modern architecture.
  - Polars over Pandas: For 10-30x faster performance and significantly lower memory usage.
  - selectolax over Beautiful Soup: For 25x faster HTML parsing with a modern engine.
  - FastAPI over Flask/Django: For built-in async support, automatic API documentation, and integrated type hinting.
  - Plotly/Altair over Matplotlib: For interactive and declarative data visualizations.

  Essential Modern Tools
  - Rich: For beautiful and informative terminal output.
  - Typer: For creating elegant and user-friendly command-line applications.
  - Pydantic: For robust and reliable data validation.
  - Ruff: For ultra-fast linting, formatting, and import sorting in one tool.
  - uv: For modern and fast package management.