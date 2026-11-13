
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from api.routes import router

app = FastAPI(
    title="Assignment - Multi-Document RAG System",
    description=(
        "A Generative AI application that answers questions "
        "using multiple uploaded PDF documents."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to Assignment - Multi-Document RAG System",
        "docs": "/docs",
    }


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Use the older binary format understood by Swagger UI.
    schema["openapi"] = "3.0.3"

    components = schema.get("components", {}).get("schemas", {})

    for component in components.values():
        properties = component.get("properties", {})

        for prop in properties.values():

            # Handle multiple file uploads.
            if prop.get("type") == "array":
                items = prop.get("items", {})

                if items.get("contentMediaType") == "application/octet-stream":
                    items.pop("contentMediaType", None)
                    items["format"] = "binary"

            # Handle single file uploads.
            elif prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
