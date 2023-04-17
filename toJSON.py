import json

def unicode_escape_to_text(escaped_str: str) -> str:
    return escaped_str.encode('utf-8').decode('unicode_escape')

def to_json_compatible_string(input_str: str) -> str:
    # 줄바꿈 문자를 처리합니다.
    input_str = input_str.replace('\r\n', '\\n').replace('\n', '\\n')

    # 이스케이프 문자를 적용합니다.
    escaped_str = json.dumps(input_str)

    # 큰따옴표를 제거합니다.
    return escaped_str[1:-1]

input_data = '여기에\n줄바꿈이 포함된\n텍스트가 있습니다.'
json_compatible_string = to_json_compatible_string(input_data)
output_string = unicode_escape_to_text(json_compatible_string)
print(output_string)
