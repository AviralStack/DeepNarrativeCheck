import pathway as pw

class BookSchema(pw.Schema):
    text: str
    book: str

def build_index(book_files):
    rows = []
    for book_name, content in book_files.items():
        rows.append({"text": content, "book": book_name})

    table = pw.debug.table_from_rows(BookSchema, rows)
    return table
