# Example input

`patient.csv` contains one invented hospital encounter for trying the prediction command.
It contains no patient identifiers and no known readmission outcome. Keep its column
names when making your own input file.

```bash
python test_model.py --input examples/patient.csv
```

Use `--results my_run` if you trained your model in another folder. See
[the testing guide](../docs/testing.md) for setup and performance checks.
