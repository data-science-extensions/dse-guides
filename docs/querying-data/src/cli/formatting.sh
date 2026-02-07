uv run lint
uv run extract-sections-from-markdown-file index.md pandas
uv run extract-sections-from-markdown-file index.md sql
uv run extract-sections-from-markdown-file index.md pyspark
uv run extract-sections-from-markdown-file index.md polars
uv run reformat-and-convert-md-to-ipynb index.md h3
uv run reformat-and-convert-md-to-ipynb index-pandas.md h3
uv run reformat-and-convert-md-to-ipynb index-sql.md h3
uv run reformat-and-convert-md-to-ipynb index-pyspark.md h3
uv run reformat-and-convert-md-to-ipynb index-polars.md h3
