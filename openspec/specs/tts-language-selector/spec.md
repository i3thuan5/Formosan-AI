### Requirement: 族別選擇使用 Radio 元件

TTS 系統的「預設配音員」與「自己當配音員」兩個 Tab 中，族別選擇 SHALL 使用 `gr.Radio` 元件呈現所有可用族別。

#### Scenario: 預設配音員 Tab 顯示族別 Radio

- **WHEN** 使用者進入「預設配音員」Tab
- **THEN** 族別選擇以 Radio 元件呈現，顯示所有 ETHNICITIES 選項，預設選取「阿美」

#### Scenario: 自訂配音員 Tab 顯示族別 Radio

- **WHEN** 使用者進入「自己當配音員」Tab
- **THEN** 族別選擇以 Radio 元件呈現，顯示所有 ETHNICITIES 選項，預設選取「阿美」

### Requirement: 配音員選擇使用 Radio 元件

「預設配音員」Tab 中，配音員選擇 SHALL 使用 `gr.Radio` 元件，依所選族別動態更新選項。

#### Scenario: 切換族別後配音員選項更新

- **WHEN** 使用者在「預設配音員」Tab 選擇不同族別
- **THEN** 配音員 Radio 的 choices 更新為該族別下所有配音員，value 自動選取第一位配音員

#### Scenario: 配音員 Radio 初始狀態

- **WHEN** 頁面載入完成
- **THEN** 配音員 Radio 顯示「阿美」族別下的所有配音員，預設選取第一位

### Requirement: 語別選擇使用 Radio 元件

「自己當配音員」Tab 中，語別選擇 SHALL 使用 `gr.Radio` 元件，依所選族別動態更新選項。

#### Scenario: 切換族別後語別選項更新

- **WHEN** 使用者在「自己當配音員」Tab 選擇不同族別
- **THEN** 語別 Radio 的 choices 更新為該族別下所有語別，value 自動選取第一個語別

#### Scenario: 單一語別時隱藏語別 Radio

- **WHEN** 使用者選擇的族別只有一個語別
- **THEN** 語別 Radio SHALL 設定 `visible=False` 隱藏元件

#### Scenario: 多語別時顯示語別 Radio

- **WHEN** 使用者選擇的族別有多於一個語別
- **THEN** 語別 Radio SHALL 設定 `visible=True` 顯示元件

### Requirement: Radio 元件不使用 filterable 參數

所有由 Dropdown 轉換為 Radio 的元件 SHALL NOT 包含 `filterable` 參數。

#### Scenario: Radio 元件建構時無 filterable

- **WHEN** Radio 元件被建構或動態更新
- **THEN** 不傳入 `filterable` 參數（`gr.Radio` 不支援此屬性）
