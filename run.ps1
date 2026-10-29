# Runs the SECOM pipeline with a local JDK 17 and an unpacked Apache Spark 4.2.1.
# Edit the two paths if yours live somewhere else.
$Jdk   = "C:\Users\Umam\jdk17\jdk-17.0.20.1+1"
$Spark = "C:\Users\Umam\spark\spark-4.2.1-bin-hadoop3"

$env:JAVA_HOME  = $Jdk
$env:SPARK_HOME = $Spark
$env:Path       = "$Jdk\bin;$env:Path"
$py4j           = (Get-ChildItem "$Spark\python\lib\py4j-*-src.zip" | Select-Object -First 1).FullName
$env:PYTHONPATH = "$Spark\python;$py4j"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$here\secom_pipeline.py"
