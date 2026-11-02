import os
import sys
import warnings

# Suppress harmless pandas / arrow version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure Spark and Py4J libraries from /opt/spark are on Python's path
if "/opt/spark/python" not in sys.path:
    os.environ["SPARK_HOME"] = "/opt/spark"
    sys.path.insert(0, "/opt/spark/python/lib/py4j-0.10.9.9-src.zip")
    sys.path.insert(0, "/opt/spark/python")

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "Student_Dropout_Pipeline") -> SparkSession:
    """
    Initializes and returns a properly configured local SparkSession.
    Configures Spark to use the local filesystem, avoiding unneeded HDFS connection attempts.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    # Set log level to WARN to reduce console clutter
    spark.sparkContext.setLogLevel("WARN")
    return spark
