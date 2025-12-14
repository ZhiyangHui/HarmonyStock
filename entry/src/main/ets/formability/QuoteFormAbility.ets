import http from '@ohos.net.http'
import preferences from '@ohos.data.preferences'
import hilog from '@ohos.hilog'
import { Want } from '@kit.AbilityKit'
import { FormExtensionAbility, formBindingData, formInfo, formProvider } from '@kit.FormKit'

interface BackendQuote {
  code: string
  name: string
  price: number
  change: number
  change_percent: number
}

interface WidgetBindData {
  formId: string
  title: string
  code: string
  name: string
  priceText: string
  changeText: string
  pctText: string
  tsText: string
  hasPrev: boolean
  hasNext: boolean
}

interface FormBindingEntry {
  formId: string
  type: 'stock' | 'index'
  codes: string[]
  cursor: number
}

export default class QuoteFormAbility extends FormExtensionAbility {
  private readonly TAG: string = 'HarmonyStockWidget'
  private readonly DOMAIN: number = 0x1300

  private readonly PREF_FILE: string = 'harmonystock_widget'
  private readonly PREF_KEY_FORM_ENTRIES: string = 'form_entries_json'

  private readonly HTTP_CONNECT_TIMEOUT: number = 55000
  private readonly HTTP_READ_TIMEOUT: number = 55000

  private readonly DEFAULT_TYPE: 'stock' | 'index' = 'stock'
  private readonly DEFAULT_CODES: string[] = ['300750', '688981', '601012']

  onAddForm(want: Want): formBindingData.FormBindingData {
    const formId: string = this.getFormIdFromWant(want)
    const type: 'stock' | 'index' = this.DEFAULT_TYPE
    const codes: string[] = this.filterCodes6(this.DEFAULT_CODES)
    const cursor: number = 0

    hilog.info(this.DOMAIN, this.TAG, `onAddForm formId=%{public}s`, formId)

    this.upsertEntry({ formId, type, codes, cursor }).catch(() => {})

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
    const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(init)

    this.refreshOneForm(formId).catch(() => {})

    return binding
  }

  onUpdateForm(formId: string): void {
    hilog.info(this.DOMAIN, this.TAG, `onUpdateForm formId=%{public}s`, formId)
    this.refreshOneForm(formId).catch(() => {})
  }

  /**
   * 关键：卡片 message 事件入口
   * 你之前“没有任何日志”，很可能是 console 在 extension 里看不到
   * 这里强制用 hilog。
   */
  onFormEvent(formId: string, message: string): void {
    hilog.info(this.DOMAIN, this.TAG, `onFormEvent formId=%{public}s msg=%{public}s`, formId, message)
    this.handleMessageEvent(formId, message).catch(() => {})
  }

  private async handleMessageEvent(formId: string, message: string): Promise<void> {
    // 方案2：只接受 next（整卡轮播）
    const m: string = message.trim()
    if (m !== 'next') {
      // 收到未知消息也刷新一下，便于你观察
      await this.refreshOneForm(formId)
      return
    }

    const entry: FormBindingEntry | null = await this.loadEntry(formId)
    if (entry === null) {
      await this.refreshOneForm(formId)
      return
    }

    const n: number = entry.codes.length
    if (n <= 1) {
      await this.refreshOneForm(formId)
      return
    }

    const nextCursor: number = (entry.cursor + 1) % n
    const updated: FormBindingEntry = {
      formId: entry.formId,
      type: entry.type,
      codes: entry.codes,
      cursor: nextCursor
    }

    hilog.info(this.DOMAIN, this.TAG, `switch cursor %d -> %d`, entry.cursor, nextCursor)

    await this.upsertEntry(updated)
    await this.refreshOneForm(formId)
  }

  private async refreshOneForm(formId: string): Promise<void> {
    const entry: FormBindingEntry | null = await this.loadEntry(formId)

    const type: 'stock' | 'index' = entry ? entry.type : this.DEFAULT_TYPE
    const codes: string[] = entry ? entry.codes : this.filterCodes6(this.DEFAULT_CODES)
    const cursor: number = entry ? this.clampCursor(entry.cursor, codes.length) : 0
    const code: string = (codes.length > 0) ? codes[cursor] : '------'

    hilog.info(this.DOMAIN, this.TAG, `refresh formId=%{public}s code=%{public}s`, formId, code)

    const q: BackendQuote | null = (code.length === 6) ? await this.fetchBackendQuote(type, code) : null

    const data: WidgetBindData = this.buildWidgetData(formId, type, codes, cursor, q)
    const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(data)
    await formProvider.updateForm(formId, binding)
  }

  private buildWidgetData(
    formId: string,
    type: 'stock' | 'index',
    codes: string[],
    cursor: number,
    q: BackendQuote | null
  ): WidgetBindData {
    const now: Date = new Date()
    const hh: string = now.getHours().toString().padStart(2, '0')
    const mm: string = now.getMinutes().toString().padStart(2, '0')

    const hasPrev: boolean = (codes.length > 1)
    const hasNext: boolean = (codes.length > 1)

    if (q === null) {
      const code: string = (codes.length > 0 && cursor >= 0 && cursor < codes.length) ? codes[cursor] : '------'
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

  private async fetchBackendQuote(type: 'stock' | 'index', code: string): Promise<BackendQuote | null> {
    const httpRequest = http.createHttp()
    try {
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`
      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      if (res.responseCode === 200 && res.result) {
        const obj: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        return obj
      }
      return null
    } catch (e) {
      return null
    } finally {
      httpRequest.destroy()
    }
  }

  private getFormIdFromWant(want: Want): string {
    if (!want.parameters) return ''
    const params: Record<string, Object> = want.parameters as Record<string, Object>

    const k1: string = formInfo.FormParam.IDENTITY_KEY
    const s1: string = this.readStringParam(params, k1)
    if (s1.length > 0) return s1

    const s2: string = this.readStringParam(params, 'ohos.extra.param.key.form_identity')
    if (s2.length > 0) return s2

    return ''
  }

  private readStringParam(params: Record<string, Object>, key: string): string {
    const v: Object | undefined = params[key]
    if (v === undefined || v === null) return ''
    if (typeof v === 'string') return (v as string).trim()
    return ''
  }

  private async upsertEntry(entry: FormBindingEntry): Promise<void> {
    if (entry.formId.length === 0) return

    const pref = await preferences.getPreferences(this.context, this.PREF_FILE)
    const jsonStr: string = (await pref.get(this.PREF_KEY_FORM_ENTRIES, '')) as string
    const list: FormBindingEntry[] = this.safeParseEntries(jsonStr)

    let replaced: boolean = false
    for (let i: number = 0; i < list.length; i++) {
      if (list[i].formId === entry.formId) {
        list[i] = entry
        replaced = true
        break
      }
    }
    if (!replaced) list.push(entry)

    await pref.put(this.PREF_KEY_FORM_ENTRIES, JSON.stringify(list))
    await pref.flush()
  }

  private async loadEntry(formId: string): Promise<FormBindingEntry | null> {
    if (formId.length === 0) return null

    const pref = await preferences.getPreferences(this.context, this.PREF_FILE)
    const jsonStr: string = (await pref.get(this.PREF_KEY_FORM_ENTRIES, '')) as string
    const list: FormBindingEntry[] = this.safeParseEntries(jsonStr)

    for (let i: number = 0; i < list.length; i++) {
      if (list[i].formId === formId) {
        return {
          formId: list[i].formId,
          type: list[i].type,
          codes: list[i].codes,
          cursor: this.clampCursor(list[i].cursor, list[i].codes.length)
        }
      }
    }
    return null
  }

  private safeParseEntries(jsonStr: string): FormBindingEntry[] {
    try {
      const raw: Object = JSON.parse(jsonStr) as Object
      if (!Array.isArray(raw)) return []

      const arr: Object[] = raw as Object[]
      const out: FormBindingEntry[] = []

      for (let i: number = 0; i < arr.length; i++) {
        const itObj: Object = arr[i]
        if (itObj === null || typeof itObj !== 'object' || Array.isArray(itObj)) continue

        const rec: Record<string, Object> = itObj as Record<string, Object>

        const formIdObj: Object | undefined = rec['formId']
        const typeObj: Object | undefined = rec['type']
        const codesObj: Object | undefined = rec['codes']
        const cursorObj: Object | undefined = rec['cursor']

        if (typeof formIdObj !== 'string' || typeof typeObj !== 'string') continue

        const formId: string = (formIdObj as string).trim()
        const typeStr: string = (typeObj as string).trim()
        if (formId.length === 0) continue
        if (typeStr !== 'stock' && typeStr !== 'index') continue
        const type: 'stock' | 'index' = (typeStr === 'stock') ? 'stock' : 'index'

        const codes: string[] = this.parseCodes6(codesObj)
        if (codes.length === 0) continue

        let cursor: number = 0
        if (typeof cursorObj === 'number') cursor = Math.floor(cursorObj as number)
        cursor = this.clampCursor(cursor, codes.length)

        out.push({ formId, type, codes, cursor })
      }

      return out
    } catch (e) {
      return []
    }
  }

  private parseCodes6(obj: Object | undefined): string[] {
    if (obj === undefined || obj === null) return []
    if (!Array.isArray(obj)) return []

    const arr: Object[] = obj as Object[]
    const out: string[] = []
    const seen: Set<string> = new Set<string>()

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

  private filterCodes6(list: string[]): string[] {
    const out: string[] = []
    const seen: Set<string> = new Set<string>()
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
