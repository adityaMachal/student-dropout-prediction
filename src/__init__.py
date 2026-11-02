"""
Predicting Student Dropout Risk Package
Distributed Machine Learning using Apache Spark 4.2.0 (PySpark MLlib)
"""
import os
import sys

# Ensure Spark and Py4J libraries from /opt/spark are on Python's path
if "/opt/spark/python" not in sys.path:
    os.environ["SPARK_HOME"] = "/opt/spark"
    sys.path.insert(0, "/opt/spark/python/lib/py4j-0.10.9.9-src.zip")
    sys.path.insert(0, "/opt/spark/python")

__version__ = "1.0.0"
