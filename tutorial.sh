python minifier_exception_inductor.py
TORCHDYNAMO_REPRO_AFTER="aot" python minifier_exception_inductor.py


python minifier_accuracy_inductor.py
TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4 python minifier_accuracy_inductor.py
