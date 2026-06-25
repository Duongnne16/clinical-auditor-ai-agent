import json
from pathlib import Path

import pytest
import requests
from bs4 import BeautifulSoup

from data_pipeline.crawlers.trungtamthuoc_ingredients_crawler import (
    CrawlError,
    INDEX_URL,
    clean_text,
    detect_suffix_chain_line,
    extract_ingredient_links,
    extract_main_content,
    extract_updated_at,
    is_ingredient_url,
    is_valid_ingredient_page,
    map_section_name,
    normalize_heading,
    normalize_url,
    parse_ingredient_page,
    parse_sections,
    parse_table,
    run_crawler,
    slug_from_url,
)


DETAIL_URL = "https://trungtamthuoc.com/hoat-chat/cetirizine"


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: int) -> FakeResponse:
        self.calls.append(url)
        result = self.responses[url]
        if isinstance(result, Exception):
            raise result
        return result


def valid_html() -> str:
    long_text = "Cetirizine điều trị dị ứng và thông tin y khoa. " * 20
    return f"""
    <html><head><title>Cetirizine - Dược thư</title></head><body>
      <header>noise</header>
      <article class="article-content">
        <h1>Cetirizine Hydrochlorid</h1>
        <p>Ngày cập nhật: 01/06/2026</p>
        <p>{long_text}</p>
        <h2>1 Dạng thuốc và hàm lượng</h2><p>Viên 10 mg.</p>
        <h2>4 Chỉ định và chống chỉ định</h2>
        <h3>4.1 Chỉ định</h3><p>Điều trị viêm mũi dị ứng.</p>
        <h3>4.2 Chống chỉ định</h3><p>Quá mẫn.</p>
        <h2>5 Thận trọng</h2><p>Thận trọng khi lái xe.</p>
        <h2>9 Tương tác thuốc</h2>
        <table><tr><th>Thuốc</th><th>Tác động</th></tr>
        <tr><td>Rượu</td><td>Tăng an thần</td></tr></table>
        <h2>Tài liệu tham khảo</h2><p>Không được lưu.</p>
      </article>
    </body></html>
    """


def malformed_nested_html() -> str:
    return """
    <article>
      <div class="author">
        <p>Dược sĩ Thảo Phương Dược sĩ lâm sàng</p>
        <p>Ước tính: 2 phút đọc, Cập nhật: 01/01/2026</p>
      </div>
      <div class="content">
        <p>Thông tin mở đầu hữu ích.
          <p><h2>2 Dược lực học</h2>
            <p>Chỉ nội dung dược lực học.
              <p><h2>3 Dược động học</h2>
                <p>Chỉ nội dung dược động học.
                  <p><h2>5 Chống chỉ định</h2>
                    <p>Không dùng khi quá mẫn.
                      <p><h2>6 Thận trọng</h2>
                        <p>Thận trọng khi lái xe.
                          <p><h2>7 Thời kỳ mang thai và cho con bú</h2>
                            <p>Tham khảo bác sĩ khi mang thai.</p>
                          </p>
                        </p>
                      </p>
                    </p>
                  </p>
                </p>
              </p>
            </p>
          </p>
        </p>
        <h2>9 Tương tác thuốc</h2>
        <ul><li>Rượu <strong>tăng an thần</strong></li></ul>
        <table><tr><td>A</td><td>B</td></tr></table>
        <h2>Tài liệu tham khảo</h2><p>Không được lưu.</p>
      </div>
    </article>
    """


def test_text_url_and_heading_helpers() -> None:
    assert clean_text(" a\xa0 \u200b b\n c ") == "a b c"
    assert normalize_url("/hoat-chat/cetirizine?x=1#top") == DETAIL_URL
    assert is_ingredient_url(DETAIL_URL)
    assert not is_ingredient_url("https://trungtamthuoc.com/hoat-chat")
    assert slug_from_url(DETAIL_URL) == "cetirizine"
    assert extract_updated_at("Ngày cập nhật: 1/6/2026 10:00") == "1/6/2026"
    assert normalize_heading("10.2 Quá liều & xử trí") == "qua_lieu_xu_tri"
    assert (
        map_section_name("4 Chỉ định và chống chỉ định")
        == "chi_dinh_chong_chi_dinh"
    )
    assert (
        map_section_name("7 Tác dụng không mong muốn (ADR)")
        == "tac_dung_khong_mong_muon"
    )


def test_discovery_prefers_actual_key_links_and_filters_urls() -> None:
    key_url = f"{INDEX_URL}?key=x"
    session = FakeSession(
        {
            INDEX_URL: FakeResponse(
                '<a href="/hoat-chat?key=x">X</a>'
                '<a href="/hoat-chat?key=x">duplicate</a>'
            ),
            key_url: FakeResponse(
                '<a href="/hoat-chat/cetirizine">Cetirizine</a>'
                '<a href="/san-pham/foo">Product</a>'
                '<a href="https://example.com/hoat-chat/external">External</a>'
            ),
        }
    )

    delays: list[bool] = []
    assert extract_ingredient_links(
        session=session,
        delay=lambda: delays.append(True),
    ) == [
        DETAIL_URL
    ]
    assert session.calls == [INDEX_URL, key_url]
    assert delays == [True]


def test_extract_main_content_and_sections() -> None:
    soup = BeautifulSoup(
        "<body><main><p>" + ("x " * 70) + "</p></main>"
        "<article><h1>A</h1><p>" + ("y " * 100) + "</p></article></body>",
        "html.parser",
    )
    assert extract_main_content(soup).name == "article"

    sections = parse_sections(
        BeautifulSoup(valid_html(), "html.parser").article
    )
    assert sections["chi_dinh"] == "Điều trị viêm mũi dị ứng."
    assert sections["chong_chi_dinh"] == "Quá mẫn."
    assert "Rượu | Tăng an thần" in sections["tuong_tac_thuoc"]
    assert "Không được lưu" not in json.dumps(sections, ensure_ascii=False)


def test_parse_standard_table_preserves_columns() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr><th>Cơ quan</th><th>Thường gặp</th><th>Hiếm</th></tr>
          <tr><td>Da</td><td>Phát ban</td><td></td></tr>
        </table>
        """,
        "html.parser",
    ).table

    assert parse_table(table) == [
        "Cơ quan | Thường gặp | Hiếm",
        "Da | Phát ban",
    ]


def test_detect_suffix_chain_line_only_flags_nested_suffixes() -> None:
    assert detect_suffix_chain_line("A B C D | B C D | C D | D")
    assert not detect_suffix_chain_line(
        "Cơ quan | Rất phổ biến | Thường gặp | Không phổ biến | Hiếm | Không rõ"
    )
    assert not detect_suffix_chain_line("Thuốc | Liều dùng | Ghi chú")


def test_single_cell_and_colspan_rows_are_not_force_split() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr><td>Ghi chú duy nhất</td></tr>
          <tr><td colspan="4"><span>Thông tin chung</span></td></tr>
        </table>
        """,
        "html.parser",
    ).table

    assert parse_table(table) == [
        "Ghi chú duy nhất",
        "Thông tin chung",
    ]


def test_hybrid_recovery_removes_aciclovir_style_suffix_chain() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr>
            <td>Độ thanh thải creatinin
              <td>Liều thông thường
                <td>Khoảng cách liều</td>
              </td>
            </td>
          </tr>
          <tr>
            <td>10-25 ml/phút
              <td>800 mg
                <td>Mỗi 8 giờ</td>
              </td>
            </td>
          </tr>
        </table>
        """,
        "html.parser",
    ).table
    diagnostics: dict[str, int] = {}

    rows = parse_table(table, diagnostics)

    assert rows == [
        "Độ thanh thải creatinin | Liều thông thường | Khoảng cách liều",
        "10-25 ml/phút | 800 mg | Mỗi 8 giờ",
    ]
    assert not any(detect_suffix_chain_line(row) for row in rows)
    assert diagnostics == {
        "row_count": 2,
        "max_pipe_count_per_row": 2,
        "suspected_suffix_chain_count": 0,
    }


def test_hybrid_recovery_preserves_acetylcystein_dose_columns() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr><th>Thời gian
            <th>Liều đầu
              <th>Liều duy trì
                <th>Đường dùng</th>
              </th>
            </th>
          </th></tr>
          <tr><td>0-1 giờ
            <td>150 mg/kg
              <td>50 mg/kg
                <td>Truyền tĩnh mạch</td>
              </td>
            </td>
          </td></tr>
        </table>
        """,
        "html.parser",
    ).table

    assert parse_table(table) == [
        "Thời gian | Liều đầu | Liều duy trì | Đường dùng",
        "0-1 giờ | 150 mg/kg | 50 mg/kg | Truyền tĩnh mạch",
    ]


def test_hybrid_recovery_preserves_abiraterone_adverse_event_columns() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr><th>Hệ cơ quan
            <th>Rất thường gặp
              <th>Thường gặp
                <th>Ít gặp</th>
              </th>
            </th>
          </th></tr>
          <tr><td>Tim mạch
            <td>Tăng huyết áp
              <td>Suy tim
                <td>Loạn nhịp</td>
              </td>
            </td>
          </td></tr>
        </table>
        """,
        "html.parser",
    ).table

    rows = parse_table(table)

    assert rows == [
        "Hệ cơ quan | Rất thường gặp | Thường gặp | Ít gặp",
        "Tim mạch | Tăng huyết áp | Suy tim | Loạn nhịp",
    ]
    assert not any(detect_suffix_chain_line(row) for row in rows)


def test_parse_malformed_nested_table_without_suffix_duplication() -> None:
    table = BeautifulSoup(
        """
        <table>
          <tr>
            <th>Cơ quan/Hệ cơ quan
              <th>Tần suất xuất hiện
                <tr>
                  <td>
                    <td>Rất phổ biến</td><td>Thường gặp</td>
                    <td>Không phổ biến</td><td>Hiếm</td><td>Không rõ</td>
                    <tr>
                      <td>Nhiễm trùng</td><td>Viêm mũi họng</td>
                      <td>Nhiễm herpes</td><td></td><td></td><td></td>
                    </tr>
                  </td>
                </tr>
              </th>
            </th>
          </tr>
        </table>
        """,
        "html.parser",
    ).table

    rows = parse_table(table)

    assert rows[0] == "Cơ quan/Hệ cơ quan | Tần suất xuất hiện"
    assert rows[1] == " | Rất phổ biến | Thường gặp | Không phổ biến | Hiếm | Không rõ"
    assert rows[2] == "Nhiễm trùng | Viêm mũi họng | Nhiễm herpes"
    assert len(rows) == 3


def test_nested_table_rows_are_not_duplicated_in_outer_table() -> None:
    soup = BeautifulSoup(
        """
        <table id="outer">
          <tr><td>Outer A</td><td><table><tr><td>Inner</td></tr></table></td></tr>
          <tr><td>Outer B</td><td>Value</td></tr>
        </table>
        """,
        "html.parser",
    )

    assert parse_table(soup.find("table", id="outer")) == [
        "Outer A",
        "Outer B | Value",
    ]


def test_parse_sections_handles_malformed_nested_paragraphs() -> None:
    sections = parse_sections(
        BeautifulSoup(malformed_nested_html(), "html.parser").article
    )

    assert sections["mo_ta_chung"] == "Thông tin mở đầu hữu ích."
    assert sections["duoc_luc_hoc"] == "Chỉ nội dung dược lực học."
    assert sections["duoc_dong_hoc"] == "Chỉ nội dung dược động học."
    assert sections["chong_chi_dinh"] == "Không dùng khi quá mẫn."
    assert sections["than_trong"] == "Thận trọng khi lái xe."
    assert sections["thai_ky_cho_con_bu"] == "Tham khảo bác sĩ khi mang thai."
    assert sections["tuong_tac_thuoc"] == "Rượu tăng an thần\nA | B"
    serialized = json.dumps(sections, ensure_ascii=False)
    assert "Dược động học" not in sections["duoc_luc_hoc"]
    assert "Thận trọng" not in sections["chong_chi_dinh"]
    assert "Thời kỳ mang thai" not in sections["than_trong"]
    assert "Không được lưu" not in serialized
    assert "Ước tính" not in sections["mo_ta_chung"]
    assert "Dược sĩ" not in sections["mo_ta_chung"]


@pytest.mark.parametrize(
    "heading",
    [
        "14 So sánh sự khác biệt giữa Paracetamol và Ibuprofen?",
        "Sự khác biệt giữa hai hoạt chất",
        "Paracetamol và Ibuprofen",
        "Bài viết liên quan",
        "Sản phẩm liên quan",
        "Câu hỏi thường gặp",
        "Tài liệu tham khảo",
        "Giải đáp các thắc mắc",
    ],
)
def test_supplementary_article_headings_stop_parsing(heading: str) -> None:
    html = f"""
    <article>
      <h2>11 Tương tác thuốc</h2><p>Nội dung tương tác.</p>
      <h2>13 Quá liều và xử trí</h2>
      <h3>13.1 Triệu chứng</h3><p>Triệu chứng quá liều.</p>
      <h3>13.2 Xử trí</h3><p>Cách xử trí.</p>
      <h2>{heading}</h2>
      <h3>Paracetamol</h3><p>Nội dung bài phụ.</p>
    </article>
    """

    sections = parse_sections(BeautifulSoup(html, "html.parser").article)

    assert sections["tuong_tac_thuoc"] == "Nội dung tương tác."
    assert sections["trieu_chung"] == "Triệu chứng quá liều."
    assert sections["xu_tri"] == "Cách xử trí."
    assert "paracetamol" not in sections
    assert "Nội dung bài phụ" not in json.dumps(
        sections, ensure_ascii=False
    )


def test_medical_headings_do_not_trigger_stop_rules() -> None:
    html = """
    <article>
      <h2>Chỉ định</h2><p>A</p>
      <h2>Chống chỉ định</h2><p>B</p>
      <h2>Thận trọng</h2><p>C</p>
      <h2>Thời kỳ mang thai</h2><p>D</p>
      <h2>Thời kỳ mang thai và cho con bú</h2><p>E</p>
      <h2>Tác dụng không mong muốn</h2><p>F</p>
      <h2>Liều lượng và cách dùng</h2><p>G</p>
      <h2>Tương tác thuốc</h2><p>H</p>
      <h2>Triệu chứng</h2><p>I</p>
      <h2>Xử trí</h2><p>J</p>
    </article>
    """

    sections = parse_sections(BeautifulSoup(html, "html.parser").article)

    assert sections["chi_dinh"] == "A"
    assert sections["chong_chi_dinh"] == "B"
    assert sections["than_trong"] == "C"
    assert sections["thoi_ky_mang_thai"] == "D"
    assert sections["thai_ky_cho_con_bu"] == "E"
    assert sections["tac_dung_khong_mong_muon"] == "F"
    assert sections["lieu_luong_va_cach_dung"] == "G"
    assert sections["tuong_tac_thuoc"] == "H"
    assert sections["trieu_chung"] == "I"
    assert sections["xu_tri"] == "J"


def test_validation_returns_debug_detail() -> None:
    valid, reason, detail = is_valid_ingredient_page(None, "short", {})

    assert not valid
    assert reason == "missing_title"
    assert detail == {
        "title": None,
        "text_length": 5,
        "medical_signal_count": 0,
        "section_count": 0,
    }


def test_parse_page_has_schema_and_crawled_at() -> None:
    record = parse_ingredient_page(
        DETAIL_URL,
        session=FakeSession({DETAIL_URL: FakeResponse(valid_html())}),
    )

    assert record["source"] == "trungtamthuoc"
    assert record["entity_type"] == "ingredient"
    assert record["name"] == "Cetirizine Hydrochlorid"
    assert record["updated_at"] == "01/06/2026"
    assert record["crawled_at"].endswith("+00:00")
    assert len(record["sections"]) >= 2


def test_parse_page_reports_status_and_timeout() -> None:
    with pytest.raises(CrawlError) as status_error:
        parse_ingredient_page(
            DETAIL_URL,
            session=FakeSession({DETAIL_URL: FakeResponse("", 503)}),
        )
    assert status_error.value.reason == "invalid_status"

    with pytest.raises(CrawlError) as timeout_error:
        parse_ingredient_page(
            DETAIL_URL,
            session=FakeSession({DETAIL_URL: requests.Timeout("slow")}),
        )
    assert timeout_error.value.reason == "timeout"


def test_streaming_run_writes_jsonl_and_failed_details(tmp_path: Path) -> None:
    second_url = "https://trungtamthuoc.com/hoat-chat/short"
    session = FakeSession(
        {
                DETAIL_URL: FakeResponse(valid_html()),
                second_url: FakeResponse(
                    "<html><body><article>"
                    + ("<p>short text</p>" * 20)
                    + "<h1>Short</h1></article></body></html>"
                ),
        }
    )
    raw_path = tmp_path / "ingredients_raw.jsonl"
    failed_path = tmp_path / "failed_urls.json"
    sleeps: list[float] = []

    success, failures, sections, reasons = run_crawler(
        [DETAIL_URL, second_url],
        session=session,
        timeout=25,
        delay_min=1.0,
        delay_max=2.5,
        raw_path=raw_path,
        failed_path=failed_path,
        overwrite=True,
        sleeper=sleeps.append,
        random_uniform=lambda low, high: 1.5,
    )

    assert success == 1
    assert sleeps == [1.5]
    assert not (tmp_path / "ingredients_raw.tmp.jsonl").exists()
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["name"] == "Cetirizine Hydrochlorid"
    assert failures[0]["reason"] == "content_too_short"
    assert failures[0]["detail"]["title"] == "Short"
    assert failures[0]["detail"]["text_length"] < 500
    assert reasons["content_too_short"] == 1
    assert sections["chi_dinh"] == 1


def test_run_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    raw_path = tmp_path / "ingredients_raw.jsonl"
    raw_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_crawler(
            [],
            session=FakeSession({}),
            timeout=25,
            delay_min=0,
            delay_max=0,
            raw_path=raw_path,
            failed_path=tmp_path / "failed_urls.json",
            overwrite=False,
        )
