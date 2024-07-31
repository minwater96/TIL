### TIL

같은 form을 갖고있는 html문서를 불러올때는 둘을 합쳐서 같이 불러올 수 있다.

block body 내부에 같은 form tap을 사용하여, 입력해주고,

상단에 if문을 사용해 상황에 맞는 html 문서를 불러줄 수 있다!

### 예시코드

```markdown
{% extends 'base.html' %}

{% block body %}

{% if request.resolver_match.url_name == 'create' %}
    <h1>create</h1>
{% else %}
    <h1>updtae</h1>
{% endif %}

<form action="" method="POST">
    {% csrf_token %}
    {{form}}
    <input type="submit">
</form>
{% endblock %}
```

#내맘대로TIL챌린지 #동아일보 #미디어 프론티어 #글로벌소프트웨어캠퍼스 #GSC신촌
글로벌소프트웨어캠퍼스와 동아일보가 함께 진행하는 챌린지입니다. 