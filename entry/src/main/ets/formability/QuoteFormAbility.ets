// 网络请求：用于从后端拉取股票 / 指数行情
import http from '@ohos.net.http'
// 本地持久化：用于保存桌面卡片的绑定状态（如轮播位置）
import preferences from '@ohos.data.preferences'
// 日志工具：用于调试 Form / 行情更新流程
import hilog from '@ohos.hilog'
// Ability / Want：桌面卡片跳转 App 时读取参数
import { Want } from '@kit.AbilityKit'
// Form 相关能力：桌面卡片（创建 / 更新 / 绑定数据）
import { FormExtensionAbility, formBindingData, formInfo, formProvider } from '@kit.FormKit'

// 后端返回的原始行情结构（字段命名按后端约定）
interface BackendQuote {
  code: string
  name: string
  price: number
  change: number
  change_percent: number
}


// 桌面卡片真正绑定的数据结构（UI 层只关心这些字符串）
interface WidgetBindData {
  formId: string          // 当前卡片实例的唯一 id
  title: string           // 卡片标题（指数 / 股票）
  code: string            // 标的代码
  name: string            // 标的名称
  priceText: string       // 当前价格（已格式化）
  changeText: string      // 涨跌额（已格式化）
  pctText: string         // 涨跌幅（已格式化）
  tsText: string          // 更新时间（如 12:30）
  hasPrev: boolean        // 是否允许“上一只”
  hasNext: boolean        // 是否允许“下一只”
}


// 每个桌面卡片在本地保存的一条“绑定记录”
interface FormBindingEntry {
  formId: string          // 桌面卡片实例 id
  type: 'stock' | 'index' // 当前卡片展示的是股票还是指数
  codes: string[]         // 可轮播的标的代码列表
  cursor: number          // 当前展示到 codes 的哪个位置
}

export default class QuoteFormAbility extends FormExtensionAbility {
  // 日志用的 TAG / DOMAIN，统一标识桌面卡片相关日志
  private readonly TAG: string = 'HarmonyStockWidget'
  private readonly DOMAIN: number = 0x1300

  // 桌面卡片本地持久化使用的首选项文件及 key
  // 用来保存每个 formId 对应的轮播状态等信息
  private readonly PREF_FILE: string = 'harmonystock_widget'
  private readonly PREF_KEY_FORM_ENTRIES: string = 'form_entries_json'

  // 行情接口请求的超时配置（和 App 侧保持一致）
  private readonly HTTP_CONNECT_TIMEOUT: number = 55000
  private readonly HTTP_READ_TIMEOUT: number = 55000

  // 桌面卡片首次创建时的默认展示类型
  private readonly DEFAULT_TYPE: 'stock' | 'index' = 'stock'

  // 桌面卡片首次创建时的默认轮播标的列表
  private readonly DEFAULT_CODES: string[] = ['300750', '688981', '601012']

  // 桌面卡片首次被添加到桌面时调用：初始化卡片状态并返回首帧数据
  onAddForm(want: Want): formBindingData.FormBindingData {
    // 从 want 中解析系统分配的 formId（每个卡片实例唯一）
    const formId: string = this.getFormIdFromWant(want)

    // 新卡片默认展示类型（股票 / 指数）
    const type: 'stock' | 'index' = this.DEFAULT_TYPE

    // 默认轮播代码列表，顺便过滤成合法的 6 位代码
    const codes: string[] = this.filterCodes6(this.DEFAULT_CODES)

    // 轮播游标，指向当前展示第几个 code
    const cursor: number = 0

    // 记录一次卡片创建日志，方便定位多卡片问题
    hilog.info(this.DOMAIN, this.TAG, `onAddForm formId=%{public}s`, formId)

    // 将该卡片的运行状态写入本地持久化（后续刷新、轮播都会依赖）
    this.upsertEntry({ formId, type, codes, cursor }).catch(() => {})

    // 构造卡片的“初始占位数据”
    // 这一步非常重要：系统要求 onAddForm 必须同步返回一份可渲染数据
    const init: WidgetBindData = {
      formId: formId,
      title: '行情',
      code: (codes.length > 0) ? codes[0] : '------',
      name: '加载中…',
      priceText: '--',
      changeText: '--',
      pctText: '--',
      tsText: '',
      hasPrev: false,
      hasNext: (codes.length > 1)
    }

    // 将业务数据包装成 FormBindingData，交给系统渲染
    const binding: formBindingData.FormBindingData =
      formBindingData.createFormBindingData(init)

    // 异步刷新真实行情（不阻塞 onAddForm 的返回）
    // 后端数据回来后会通过 updateForm 覆盖当前占位内容
    this.refreshOneForm(formId).catch(() => {})

    // 返回首帧数据给系统（同步）
    return binding
  }

  // 系统触发卡片刷新时回调（定时刷新 / 手动刷新 / 系统调度）
  onUpdateForm(formId: string): void {
    // 记录一次刷新日志，方便排查是哪张卡片在更新
    hilog.info(this.DOMAIN, this.TAG, `onUpdateForm formId=%{public}s`, formId)

    // 异步拉取最新行情并更新该卡片
    // 不阻塞回调本身，避免影响系统刷新流程
    this.refreshOneForm(formId).catch(() => {})
  }


  // 桌面卡片上的交互事件回调（例如点击“上一只 / 下一只”等按钮）
  onFormEvent(formId: string, message: string): void {
    // 打一条日志，记录是哪张卡片、触发了什么事件
    hilog.info(
      this.DOMAIN,
      this.TAG,
      `onFormEvent formId=%{public}s msg=%{public}s`,
      formId,
      message
    )

    // 根据 message 内容处理具体交互逻辑（切换标的、刷新等）
    // 这里异步执行，不阻塞系统的事件分发
    this.handleMessageEvent(formId, message).catch(() => {})
  }


  // 处理桌面卡片发来的交互消息（目前只支持“下一只”轮播）
  private async handleMessageEvent(formId: string, message: string): Promise<void> {
    // 当前实现只关心 next，其它消息统一当成一次普通刷新
    const m: string = message.trim()
    if (m !== 'next') {
      // 收到未知消息也刷新一下，方便调试和兜底
      await this.refreshOneForm(formId)
      return
    }

    // 从本地持久化中读取这张卡片对应的运行状态
    const entry: FormBindingEntry | null = await this.loadEntry(formId)
    if (entry === null) {
      // 状态丢失或异常时，直接刷新当前卡片
      await this.refreshOneForm(formId)
      return
    }

    // 当前卡片可轮播的标的数量
    const n: number = entry.codes.length
    if (n <= 1) {
      // 只有 0 或 1 个标的，轮播没有意义，直接刷新即可
      await this.refreshOneForm(formId)
      return
    }

    // 计算下一个游标位置（循环轮播）
    const nextCursor: number = (entry.cursor + 1) % n

    // 生成更新后的卡片状态（只变 cursor）
    const updated: FormBindingEntry = {
      formId: entry.formId,
      type: entry.type,
      codes: entry.codes,
      cursor: nextCursor
    }

    // 记录一次轮播切换日志
    hilog.info(
      this.DOMAIN,
      this.TAG,
      `switch cursor %d -> %d`,
      entry.cursor,
      nextCursor
    )

    // 先持久化新的游标位置
    await this.upsertEntry(updated)

    // 再根据最新状态刷新卡片内容
    await this.refreshOneForm(formId)
  }


  // 刷新指定 formId 的桌面卡片：读本地状态 -> 拉最新行情 -> 回写卡片展示数据
  private async refreshOneForm(formId: string): Promise<void> {
    // 1) 先从 preferences 里取出这张卡片的配置/运行状态（type、codes、cursor）
    const entry: FormBindingEntry | null = await this.loadEntry(formId)

    // 2) 兜底：如果 entry 不存在，就用默认类型/默认代码列表
    const type: 'stock' | 'index' = entry ? entry.type : this.DEFAULT_TYPE
    const codes: string[] = entry ? entry.codes : this.filterCodes6(this.DEFAULT_CODES)

    // 3) 兜底：cursor 可能越界（比如 codes 变短了），这里把它修正到合法范围
    const cursor: number = entry ? this.clampCursor(entry.cursor, codes.length) : 0

    // 4) 根据 cursor 选出当前要显示的 code（codes 为空则给占位符）
    const code: string = (codes.length > 0) ? codes[cursor] : '------'

    // 记录一下这次刷新到底显示哪一个，排查轮播/刷新问题时很好用
    hilog.info(this.DOMAIN, this.TAG, `refresh formId=%{public}s code=%{public}s`, formId, code)

    // 5) 只有 code 合法（6 位）才去后端拉行情；否则直接走“无数据”展示
    const q: BackendQuote | null =
      (code.length === 6) ? await this.fetchBackendQuote(type, code) : null

    // 6) 把行情 + 配置拼成桌面卡片需要的绑定数据结构（文本都在这里定）
    const data: WidgetBindData = this.buildWidgetData(formId, type, codes, cursor, q)

    // 7) 转成 FormKit 的 bindingData，并回写更新到系统桌面卡片
    const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(data)
    await formProvider.updateForm(formId, binding)
  }


  // 根据当前卡片状态 + 行情数据，组装桌面卡片真正要展示的绑定数据
  private buildWidgetData(
    formId: string,
    type: 'stock' | 'index',
    codes: string[],
    cursor: number,
    q: BackendQuote | null
  ): WidgetBindData {
    // 取当前时间，用于卡片右下角的更新时间显示
    const now: Date = new Date()
    const hh: string = now.getHours().toString().padStart(2, '0')
    const mm: string = now.getMinutes().toString().padStart(2, '0')

    // 是否显示“上一只 / 下一只”按钮
    // 当前逻辑：只要 codes 超过 1 个，就允许轮播
    const hasPrev: boolean = (codes.length > 1)
    const hasNext: boolean = (codes.length > 1)

    // 情况一：行情拉取失败（q === null）
    // 这里仍然要返回一份完整的 WidgetBindData，保证卡片可渲染
    if (q === null) {
      // 尽量显示当前 code；如果 cursor 异常就用占位符
      const code: string =
        (codes.length > 0 && cursor >= 0 && cursor < codes.length)
          ? codes[cursor]
          : '------'

      return {
        formId: formId,
        title: (type === 'index') ? '指数' : '股票',
        code: code,
        name: '网络异常',
        priceText: '--',
        changeText: '--',
        pctText: '--',
        tsText: `${hh}:${mm}`,
        hasPrev: hasPrev,
        hasNext: hasNext
      }
    }

    // 情况二：行情正常，直接用后端返回的数据填充卡片
    return {
      formId: formId,
      title: (type === 'index') ? '指数' : '股票',
      code: q.code,
      name: q.name,
      priceText: q.price.toFixed(2),
      changeText: q.change.toFixed(2),
      pctText: q.change_percent.toFixed(2) + '%',
      tsText: `${hh}:${mm}`,
      hasPrev: hasPrev,
      hasNext: hasNext
    }
  }


  // 从后端拉取单个标的的最新行情（指数 / 股票通用）
  private async fetchBackendQuote(
    type: 'stock' | 'index',
    code: string
  ): Promise<BackendQuote | null> {
    // 每次请求单独创建 http 实例，用完就销毁，避免资源泄漏
    const httpRequest = http.createHttp()
    try {
      // 按类型和代码查询单个标的行情
      const url: string =
        `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`

      // 发起 GET 请求，超时时间使用统一配置
      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      // 成功且有返回体时，解析为 BackendQuote
      if (res.responseCode === 200 && res.result) {
        const obj: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        return obj
      }

      // 非 200 或无结果，统一视为失败
      return null
    } catch (e) {
      // 网络异常 / 解析异常等，兜底返回 null，由上层决定如何展示
      return null
    } finally {
      // 无论成功失败，都要销毁 http 实例
      httpRequest.destroy()
    }
  }


  // 从 Want 中解析出桌面卡片的 formId（卡片唯一标识）
  private getFormIdFromWant(want: Want): string {
    // 如果 want 没有 parameters，说明不是卡片相关的启动/回调
    if (!want.parameters) return ''

    // parameters 在 ArkTS 里是 Object，需要显式转成可索引的字典
    const params: Record<string, Object> =
      want.parameters as Record<string, Object>

    // 官方推荐的卡片 formId 参数 key
    const k1: string = formInfo.FormParam.IDENTITY_KEY
    const s1: string = this.readStringParam(params, k1)

    // 如果通过官方 key 读到了，直接返回
    if (s1.length > 0) return s1

    // 兼容性兜底：部分系统/场景下使用的旧 key
    const s2: string = this.readStringParam(
      params,
      'ohos.extra.param.key.form_identity'
    )
    if (s2.length > 0) return s2

    // 两种 key 都没读到，说明这次不是卡片事件
    return ''
  }

  // 从参数字典中安全读取 string 类型的值
  // 任何异常 / 类型不匹配，统一返回空字符串
  private readStringParam(
    params: Record<string, Object>, // Want.parameters 转成的“安全字典”
    key: string                     // 要读取的参数名
  ): string {

    // 从字典中取值（可能不存在）
    const v: Object | undefined = params[key]

    // key 不存在，或者值为 null
    if (v === undefined || v === null) {
      return ''
    }

    // 只有当值确实是 string 时才接受
    if (typeof v === 'string') {
      return (v as string).trim()
    }

    // 其他类型（number / boolean / object / array 等）一律忽略
    return ''
  }

  // 新增或更新（upsert）一个桌面卡片绑定记录 如果同 formId 已存在 → 覆盖  如果不存在 → 追加  最终统一持久化到 preferences 中
  private async upsertEntry(entry: FormBindingEntry): Promise<void> {
    // formId 为空直接忽略，避免写入脏数据
    if (entry.formId.length === 0) return

    // 打开 / 获取本地 preferences 文件（用于桌面卡片持久化）
    const pref = await preferences.getPreferences(this.context, this.PREF_FILE)

    // 读取已保存的 entry 列表（JSON 字符串）
    const jsonStr: string =
      (await pref.get(this.PREF_KEY_FORM_ENTRIES, '')) as string

    // 安全解析为 FormBindingEntry[]（解析失败会返回空数组）
    const list: FormBindingEntry[] = this.safeParseEntries(jsonStr)

    // 标记是否发生了“替换”
    let replaced: boolean = false

    // 遍历已有列表，查找是否存在相同 formId
    for (let i: number = 0; i < list.length; i++) {
      if (list[i].formId === entry.formId) {
        // 找到：直接覆盖原 entry
        list[i] = entry
        replaced = true
        break
      }
    }

    // 如果没有找到相同 formId，说明是新卡片 → 追加
    if (!replaced) {
      list.push(entry)
    }

    // 将更新后的列表整体写回 preferences
    await pref.put(this.PREF_KEY_FORM_ENTRIES, JSON.stringify(list))
    await pref.flush() // 立即落盘，保证数据不丢
  }

  // 从 preferences 中加载指定 formId 对应的桌面卡片绑定记录（找不到返回 null）
  private async loadEntry(formId: string): Promise<FormBindingEntry | null> {
    // formId 为空直接返回，避免无意义 IO
    if (formId.length === 0) return null

    // 打开桌面卡片使用的 preferences 文件
    const pref = await preferences.getPreferences(this.context, this.PREF_FILE)

    // 读取保存的 JSON 字符串
    const jsonStr: string =
      (await pref.get(this.PREF_KEY_FORM_ENTRIES, '')) as string

    // 安全解析为 FormBindingEntry 数组（解析失败返回空数组）
    const list: FormBindingEntry[] = this.safeParseEntries(jsonStr)

    // 遍历查找对应 formId 的 entry
    for (let i: number = 0; i < list.length; i++) {
      if (list[i].formId === formId) {
        // 找到后返回一个“干净副本”
        // ⚠️ cursor 必须做边界修正，防止 codes 变化后越界
        return {
          formId: list[i].formId,
          type: list[i].type,
          codes: list[i].codes,
          cursor: this.clampCursor(
            list[i].cursor,
            list[i].codes.length
          )
        }
      }
    }

    // 没找到该 formId
    return null
  }

  // 安全解析桌面卡片绑定记录 JSON，返回合法的 FormBindingEntry 列表
  private safeParseEntries(jsonStr: string): FormBindingEntry[] {
    try {
      // 解析 JSON 字符串为原始对象
      const raw: Object = JSON.parse(jsonStr) as Object
      // 根结构必须是数组
      if (!Array.isArray(raw)) return []

      const arr: Object[] = raw as Object[]
      const out: FormBindingEntry[] = []

      // 逐条解析并校验每个 entry
      for (let i: number = 0; i < arr.length; i++) {
        const itObj: Object = arr[i]
        // 必须是普通对象，不能是 null / array
        if (itObj === null || typeof itObj !== 'object' || Array.isArray(itObj)) continue

        const rec: Record<string, Object> = itObj as Record<string, Object>

        // 读取各字段的原始值
        const formIdObj: Object | undefined = rec['formId']
        const typeObj: Object | undefined = rec['type']
        const codesObj: Object | undefined = rec['codes']
        const cursorObj: Object | undefined = rec['cursor']

        // formId / type 必须是 string
        if (typeof formIdObj !== 'string' || typeof typeObj !== 'string') continue

        const formId: string = (formIdObj as string).trim()
        const typeStr: string = (typeObj as string).trim()
        // 基本合法性校验
        if (formId.length === 0) continue
        if (typeStr !== 'stock' && typeStr !== 'index') continue
        const type: 'stock' | 'index' = (typeStr === 'stock') ? 'stock' : 'index'

        // 解析并过滤 6 位代码列表
        const codes: string[] = this.parseCodes6(codesObj)
        if (codes.length === 0) continue

        // 解析并修正 cursor，保证不越界
        let cursor: number = 0
        if (typeof cursorObj === 'number') cursor = Math.floor(cursorObj as number)
        cursor = this.clampCursor(cursor, codes.length)

        // 推入合法 entry
        out.push({ formId, type, codes, cursor })
      }

      return out
    } catch (e) {
      // JSON 解析失败或结构异常时返回空数组
      return []
    }
  }

  // 从未知对象中解析并去重得到合法的 6 位代码字符串数组
  private parseCodes6(obj: Object | undefined): string[] {
    // 为空或非数组直接返回空结果
    if (obj === undefined || obj === null) return []
    if (!Array.isArray(obj)) return []

    const arr: Object[] = obj as Object[]
    const out: string[] = []
    const seen: Set<string> = new Set<string>() // 用于去重

    // 逐项校验并收集合法的 6 位字符串
    for (let i: number = 0; i < arr.length; i++) {
      const v: Object = arr[i]
      if (typeof v !== 'string') continue
      const s: string = (v as string).trim()
      if (s.length !== 6) continue
      if (seen.has(s)) continue
      seen.add(s)
      out.push(s)
    }
    return out
  }


  // 从字符串列表中过滤并去重，得到合法的 6 位代码数组
  private filterCodes6(list: string[]): string[] {
    const out: string[] = []
    const seen: Set<string> = new Set<string>()

    // 逐个处理输入列表，保留长度为 6 的唯一代码
    for (let i: number = 0; i < list.length; i++) {
      const s: string = list[i].trim()
      if (s.length !== 6) continue
      if (seen.has(s)) continue
      seen.add(s)
      out.push(s)
    }
    return out
  }


  private clampCursor(cursor: number, n: number): number {
    if (n <= 0) return 0
    if (cursor < 0) return 0
    if (cursor >= n) return 0
    return cursor
  }
}
