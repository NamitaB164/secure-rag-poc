from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from secure_rag.ingestion.elements import ContentElement


def parse_pdf(
    path: Path,
    asset_dir: Path | None = None,
) -> list[ContentElement]:
    if not path.exists():
        raise FileNotFoundError(path)

    document = pymupdf.open(path)
    elements: list[ContentElement] = []

    for page_number, page in enumerate(document, start=1):
        tables = page.find_tables()

        for table in tables.tables:
            rows = table.extract()

            content = "\n".join(
                " | ".join(cell or "" for cell in row)
                for row in rows
            )

            elements.append(
                ContentElement(
                    type="table",
                    content=content,
                    metadata={
                        "page": page_number,
                        "rows": len(rows),
                        "columns": max(
                            (len(row) for row in rows),
                            default=0,
                        ),
                    },
                )
            )

        for image_index, image in enumerate(page.get_images(full=True)):
            xref = image[0]
            image_info = document.extract_image(xref)

            metadata = {
                "page": page_number,
                "image_index": image_index,
                "width": image_info["width"],
                "height": image_info["height"],
                "extension": image_info["ext"],
                "xref": xref,
            }

            if asset_dir is not None:
                document_asset_dir = asset_dir / path.stem
                document_asset_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                asset_path = (
                    document_asset_dir
                    / f"page-{page_number}-image-{image_index}"
                    f".{image_info['ext']}"
                )

                asset_path.write_bytes(image_info["image"])
                metadata["asset_path"] = str(asset_path)

            elements.append(
                ContentElement(
                    type="image",
                    content="",
                    metadata=metadata,
                )
            )

        text_blocks = page.get_text("blocks")

        for block in text_blocks:
            x0, y0, x1, y1, text = block[:5]

            inside_table = any(
                table.bbox[0] <= x0
                and table.bbox[1] <= y0
                and table.bbox[2] >= x1
                and table.bbox[3] >= y1
                for table in tables.tables
            )

            if text.strip() and not inside_table:
                elements.append(
                    ContentElement(
                        type="text",
                        content=text,
                        metadata={
                            "page": page_number,
                        },
                    )
                )

        page_has_text = any(
            element.type == "text"
            and element.metadata.get("page") == page_number
            for element in elements
        )

        page_has_table = any(
            element.type == "table"
            and element.metadata.get("page") == page_number
            for element in elements
        )

        if not page_has_text and not page_has_table:
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
            )

            image = Image.open(
                BytesIO(pixmap.tobytes("png")),
            )

            ocr_text = pytesseract.image_to_string(image)

            if ocr_text.strip():
                elements.append(
                    ContentElement(
                        type="text",
                        content=ocr_text,
                        metadata={
                            "page": page_number,
                            "source": "ocr",
                        },
                    )
                )

    document.close()

    return elements