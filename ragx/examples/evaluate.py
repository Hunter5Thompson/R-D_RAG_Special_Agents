"""Small evaluation script for RAGX."""

from ragx.api import public_api

if __name__ == "__main__":
    documents = [
        "Solar energy adoption accelerated in Europe between 2019 and 2024 due to subsidies.",
        "Wind power complements solar by providing generation during night hours.",
        "Grid storage projects balance renewable variability and ensure reliability.",
    ]
    public_api._reset_state_for_tests()
    public_api.index_documents(documents)
    result = public_api.answer("How do solar and wind complement each other?")
    print(result.model_dump())
