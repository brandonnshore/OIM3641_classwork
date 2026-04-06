# mcp_server.py
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP('Iris Classifier')

@mcp.tool()
def classify_iris(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
) -> str:
    """Classify an iris flower given its four measurements.
    Returns the predicted species: setosa, versicolor, or virginica.
    """
    response = httpx.post(
        'http://localhost:8000/predict',
        json={
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
    )
    result = response.json()
    return f"Predicted species: {result['species']}"

if __name__ == '__main__':
    mcp.run()
