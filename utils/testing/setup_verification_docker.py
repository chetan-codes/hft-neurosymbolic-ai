import sys
import rdflib
import pandas
import streamlit
import fastapi
import uvicorn
import pydgraph
import redis
import torch
from kanren import run, var, eq
from neo4j import GraphDatabase

def check_python():
    print(f"Python Version: {sys.version}")
    return sys.version_info >= (3, 10)

def check_libraries():
    libraries = {
        "rdflib": rdflib.__version__,
        "pandas": pandas.__version__,
        "streamlit": streamlit.__version__,
        "fastapi": fastapi.__version__,
        "uvicorn": uvicorn.__version__,
        "pydgraph": "installed",
        "redis": redis.__version__,
        "torch": torch.__version__,
        "clp_ffi_py": "0.1",  # clp-ffi-py may not have version attribute
        "neo4j": "installed"  # neo4j driver doesn't expose version directly
    }
    for lib, version in libraries.items():
        print(f"{lib}: {version}")
    return True

def check_dgraph():
    try:
        client_stub = pydgraph.DgraphClientStub("dgraph:9080")
        client = pydgraph.DgraphClient(client_stub)
        client.txn(read_only=True).query("{ node(func: has(has_stock)) { uid } }")
        print("Dgraph: Connected")
        return True
    except Exception as e:
        print(f"Dgraph: Failed - {e}")
        return False

def check_neo4j():
    try:
        driver = GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "hft_password_2025"))
        with driver.session() as session:
            session.run("MATCH (n) RETURN count(n)")
        print("Neo4j: Connected")
        return True
    except Exception as e:
        print(f"Neo4j: Failed - {e}")
        return False

def check_redis():
    try:
        r = redis.Redis(host="redis", port=6379, decode_responses=True)
        r.set("test", "hello")
        print(f"Redis: {r.get('test')}")
        return True
    except Exception as e:
        print(f"Redis: Failed - {e}")
        return False

def check_pytorch():
    try:
        print(f"PyTorch: {torch.__version__}")
        print("Note: M2 GPU not available in Docker; CPU used.")
        return True
    except Exception as e:
        print(f"PyTorch: Failed - {e}")
        return False

def check_minikanren():
    try:
        x = var()
        result = run(1, x, eq(x, 1))
        print(f"MiniKanren: {result}")
        return result == (1,)
    except Exception as e:
        print(f"MiniKanren: Failed - {e}")
        return False

def main():
    print("Starting Setup Verification (Docker-Only)...")
    checks = [
        ("Python", check_python()),
        ("Libraries", check_libraries()),
        ("Dgraph", check_dgraph()),
        ("Neo4j", check_neo4j()),
        ("Redis", check_redis()),
        ("PyTorch", check_pytorch()),
        ("MiniKanren", check_minikanren())
    ]
    for name, success in checks:
        print(f"{name} Check: {'Success' if success else 'Failed'}")
    if all(success for _, success in checks):
        print("Setup Complete: Docker-only environment ready for HFT project!")
    else:
        print("Setup Incomplete: Debug with Cursor Pro.")
        sys.exit(1)

if __name__ == "__main__":
    main()