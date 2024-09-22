from notion_client import Client
from dateutil.parser import parse
from collections import defaultdict
import requests
from pathlib import Path

import os
import json
import calendar
import datetime
from datetime import datetime, timedelta

# Notion接続情報の設定
def connect_notion(NOTION_TOKEN, NOTION_VERSION):
    notion_client = Client(auth=NOTION_TOKEN)
    notion_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }
    return notion_client, notion_headers

# 〇で囲まれた数値を・に変換する関数
def circlenum_to_dots(text):
    # Unicodeのエンコーディング範囲を使用して、①から⑳までの文字を検出し、・に置換
    for i in range(1, 21):
        text = text.replace(chr(0x245F + i), "・")
    return text

# NotionページIDを指定してページを取得
def get_page(client, page_id):
    page = client.pages.retrieve(page_id=page_id)
    return page

# ページIDを指定してタイトルを取得
def get_title_by_id(pages, page_id, item):
    for page in pages:
        if page['id'] == page_id:
            return page['properties'][item]['title'][0]['text']['content']

# ページIDを指定して数値項目を取得
def get_num_by_id(pages, page_id, item):
    for page in pages:
        if page['id'] == page_id:
            return page["properties"][item]["number"]

# ページIDを指定して日付項目を取得
def get_date_by_id(pages, page_id, item):
    date = ""
    for page in pages:
        if page['id'] == page_id and item in page['properties'] and page['properties'][item]['date']:
            date = page['properties'][item]['date']['start'][:10]
            return date

# ページIDを指定してリッチテキスト項目を取得
def get_rich_text_by_id(pages, page_id, item):
    for page in pages:
        if page['id'] == page_id:
            return page['properties'][item]['rich_text']

# ページIDを指定してリッチテキスト項目の文字列（フォント設定等を除いたもの）を取得
def get_rich_text_item_by_id(pages, page_id, item):
    rich_texts = get_rich_text_by_id(pages, page_id, item)
    content = ''.join([text['text']['content'] for text in rich_texts])
    return content

# ページIDを指定してチェック項目を取得
def get_chk_by_id(pages, page_id, item):
    for page in pages:
        if page['id'] == page_id:
            return page["properties"][item]["checkbox"]

# ページIDを指定して評価結果を取得
def get_select_by_id(pages, page_id, item):
    for page in pages:
        if page['id'] == page_id:
            return page['properties'][item]['select']['name']

# ページIDを指定してURLを取得
def get_url_by_id(pages, page_id, item):
    for page in pages:
        if page['id'] == page_id:
            if page["properties"][item]["url"]:
                url = page["properties"][item]["url"]
            else:
                url = ""
            return url

# ページIDを指定してアイテムのデータ型によって値を取得
def get_notion_item_by_id(pages, page_id, item, item_type):
    item_data = ""
    if item_type == "title":
        item_data = get_title_by_id(pages, page_id, item)
    elif item_type == "num":
        item_data = get_num_by_id(pages, page_id, item)
    elif item_type == "date":
        item_data = get_date_by_id(pages, page_id, item)
    elif item_type == "rich_text":
        item_data = get_rich_text_item_by_id(pages, page_id, item)
    elif item_type == "chk":
        item_data = get_chk_by_id(pages, page_id, item)
    elif item_type == "select":
        item_data = get_select_by_id(pages, page_id, item)
    elif item_type == "url":
        item_data = get_url_by_id(pages, page_id, item)
    else:
        item_data = "item_type" #NotionDBに無い固定値の場合はSettingファイルの記載文字列を設定
    if isinstance(item_data, str):
        item_data = circlenum_to_dots(item_data)
    return item_data

# Notionデータベースから全ページを取得
def get_all_page_ids(headers, database_id):
    has_more = True
    next_cursor = None
    all_results = []

    while has_more:
        payload = {}
        if next_cursor:
            payload["start_cursor"] = next_cursor

        response = requests.post(f"https://api.notion.com/v1/databases/{database_id}/query", headers=headers, json=payload)
        response_json = response.json()

        if response.status_code != 200:
            print(f"Error fetching page IDs. Response: {response_json}")
            return []

        all_results.extend(response_json["results"])
        has_more = response_json.get("has_more", False)
        next_cursor = response_json.get("next_cursor")

    return [result["id"] for result in all_results]

# Notionデータベースから全ページを取得
def get_all_pages(headers, database_id):
    has_more = True
    next_cursor = None
    all_results = []

    while has_more:
        payload = {}
        if next_cursor:
            payload["start_cursor"] = next_cursor

        response = requests.post(f"https://api.notion.com/v1/databases/{database_id}/query", headers=headers, json=payload)
        response_json = response.json()

        if response.status_code != 200:
            print(f"Error fetching page IDs. Response: {response_json}")
            return []

        all_results.extend(response_json["results"])
        has_more = response_json.get("has_more", False)
        next_cursor = response_json.get("next_cursor")

    return all_results

# Notionデータベースから確定済ページを取得
def get_pages_done(headers, database_id, chk_dict=None, date_dict=None):
    has_more = True
    next_cursor = None
    all_results = []
    
    while has_more:
        payload = {}
        if chk_dict is not None or date_dict is not None:
            payload = {"filter": {"and": []}}
            if chk_dict is not None:
                for chk_item, chk in chk_dict.items():
                    payload["filter"]["and"].append({
                        "property": chk_item,
                        "checkbox": {
                            "equals": chk
                        }
                    })
            if date_dict is not None:
                for date_item, set_date in date_dict.items():
                    start_date = datetime.strptime(set_date[0], "%Y-%m-%d") if set_date[0] else datetime.strptime("1000-01-01", "%Y-%m-%d")
                    end_date = datetime.strptime(set_date[1], "%Y-%m-%d") if set_date[1] else datetime.strptime("9999-12-31", "%Y-%m-%d")
                    payload["filter"]["and"].append({
                        "and": [
                            {
                                "property": date_item,
                                "date": {
                                    "on_or_after": start_date.isoformat()
                                }
                            },
                            {
                                "property": date_item,
                                "date": {
                                    "on_or_before": end_date.isoformat()
                                }
                            }
                        ]
                    })
        if next_cursor:
            payload["start_cursor"] = next_cursor
        response = requests.post(f"https://api.notion.com/v1/databases/{database_id}/query", headers=headers, json=payload)
        response_json = response.json()
        if response.status_code != 200:
            print(f"Error fetching page IDs. Response: {response_json}")
            return []
        all_results.extend(response_json["results"])
        has_more = response_json.get("has_more", False)
        next_cursor = response_json.get("next_cursor")
    return all_results

# Notionページへのデータ上書き
def update_notion_property(headers, page_id, property_name, new_value):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            property_name: {
                "rich_text": [{
                    "text": {
                        "content": new_value
                    }
                }]
            }
        }
    }
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# Notionページへのデータ上書き(リッチテキスト)
def update_notion_rich_text(headers, page_id, property_name, new_value):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            property_name: {
                "rich_text":  new_value
            }
        }
    }
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# Notionページのデータ上書き(数値)
def update_notion_num(headers, page_id, property_name, new_value):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            property_name: {
                "number": new_value
            }
        }
    }
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# Notionページのデータ上書き(選択)
def update_notion_select(headers, page_id, property_name, new_value):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            property_name: {
                "select": {
                    "name": new_value
                }
            }
        }
    }
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# Notionページのデータ上書き(日付)
def update_notion_date(headers, page_id, date_name, date):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            date_name: {
                "date": {
                    "start": date,
                    "end": None,
                    "time_zone": None
                }
            }
        }
    }  
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# NotionページのURL更新
def update_notion_url(headers, page_id, property_name, item):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            property_name: {
                "url": item,
            }
        }
    }  
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# Notionページの任意のChk項目の設定
def update_notion_chk(headers, page_id, chk_item, chk=True):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    data = {
        "properties": {
            chk_item: {
                "checkbox": chk
            }
        }
    }  
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 200:
        return "Updated successfully"
    else:
        return f"Failed to update. Reason: {response.text}"

# Notionページへのデータ上書き（ブロック）
def update_notion_block(headers, page_id, text):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    data = {
        "children": [{
            "paragraph": {
                "rich_text": [{
                    "text": {
                        "content": f"{text}"
                        }
                    }]
                }
            },]
        }
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Failed to update. Reason: {response.text}")

# Notionページへのデータ上書き（ブロック）
def update_notion_block_bold(headers, page_id, text):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    data = {
        "children": [{
            "paragraph": {
                "rich_text": [{
                    "text": {
                        "content": f"{text}"
                        },"annotations": {"bold": True,"italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"},
                    }]
                }
            },]
        }
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Failed to update. Reason: {response.text}")

# Notionデータベースからデジタルツイン生成の対象データを取得
def get_docs_dtwin(client, headers, database_id, item_dict=None, chk_dict=None, date_dict=None):  
    page_docs = []
    pages = get_pages_done(headers, database_id, chk_dict, date_dict)
    page_ids = [page['id'] for page in pages]
    for page_id in page_ids:
        if item_dict is not None:
            page_items = {}
            page_items.update({'id': page_id})
            for key, subdict in item_dict.items():
                if len(subdict) == 1:
                    for sub_key, sub_value in subdict.items():
                        if sub_value is not None:
                            page_items[key] = get_notion_item_by_id(pages, page_id, sub_key, sub_value)
                        else:
                            page_items[key] = sub_value
                elif len(subdict) > 1:
                    subitem_dict = {}
                    for sub_key, sub_value in subdict.items():
                        if sub_value is not None:
                            subitem_dict[sub_key] = get_notion_item_by_id(pages, page_id, sub_key, sub_value)
                        page_items[key] = subitem_dict
                else:
                    page_items[key] = subdict
            page_docs.append(page_items)
    return page_docs


# Notionデータベースからページの集合を与えてデジタルツインの対象データを取得
def get_docs_dtwin_pages(client, headers, database_id, pages):
    page_docs = []
    for page in pages:
        page_doc = []
        try:
            page_doc.append(page['properties']["考察生成日"]['date']['start'][:10]+"時の知識\n")
            page_doc.append(page["properties"]["名前"]["title"][0]["text"]["content"]+"\n")
            page_doc.append(page["properties"]["考察_確定稿"]["rich_text"][0]["text"]["content"])
            page_docs.append(' '.join(page_doc))
        except IndexError:
            pass
    return page_docs

# NotionページIDを指定してアーカイブ
def archive_page(headers, page_id):
    update_payload = {
        "archived": True
    }
    response = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=headers, json=update_payload)
    if response.status_code != 200:
        print(f"Error archiving page {page_id}. Response: {response.json()}")

# Notionページ追加処理
def create_page(client, headers, db_id, page_title, title_item="名前"):
    page_content = []

    # 名前が重複する記事はアーカイブ
    page_ids = get_all_page_ids(headers, db_id)
    if page_ids:
        for page_id in page_ids:
            entry_title = get_page_title(client, page_id)
            if entry_title == page_title:
                archive_page(headers, page_id)
                print(entry_title+"をアーカイブしました")

    # エントリ詳細を追加
    page_content.append({
        "object": 'block',
        "type": 'paragraph',
        "paragraph": {
            "rich_text": [{
                "text": {
                    "content": " "
                    },"annotations": {"bold": True,"italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"},
                }],
            }
        },)

    # Notionに新しいページを追加
    payload = {
        "parent": {"database_id": db_id},
        "properties": {
            title_item: {
                "title": [{
                    "text": {
                        "content": page_title
                        },
                    }],
                },
            },
        "children": page_content,
        }
    response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
    response_json = response.json()
    print(page_title+"を作成しました")

    if response.status_code != 200:
        print(f"エラーが発生しました。 {title}: {response_json}")
    return response_json