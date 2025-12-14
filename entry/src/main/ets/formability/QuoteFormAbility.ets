import http from '@ohos.net.http'
import preferences from '@ohos.data.preferences'
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
}

// 用“数组条目”代替 {formId: "..."} 这种对象映射，规避 untyped-obj-literals
interface FormBindingEntry {
  formId: string
  type: 'stock' | 'index'
  code: string
}

export default class QuoteFormAbility extends FormExtensionAbility {
  private readonly PREF_FILE: string = 'harmonystock_widget'
  private readonly PREF_KEY_FORM_ENTRIES: string = 'form_entries_json'

  private readonly HTTP_CONNECT_TIMEOUT: number = 55000
  private readonly HTTP_READ_TIMEOUT: number = 55000

  private readonly DEFAULT_TYPE: 'stock' | 'index' = 'stock'
  private readonly DEFAULT_CODE: string = '300750'

  onAddForm(want: Want): formBindingData.FormBindingData {
    const formId: string = this.getFormIdFromWant(want)
    const type: 'stock' | 'index' = this.DEFAULT_TYPE
    const code: string = this.DEFAULT_CODE

    // 1) 保存 formId -> (type, code)
    this.saveFormBinding(formId, type, code).catch(() => {})

    // 2) 先给占位数据
    const init: WidgetBindData = {
      formId: formId,
      title: '行情',
      code: code,
      name: '加载中…',
      priceText: '--',
      changeText: '--',
      pctText: '--',
      tsText: ''
    }
    const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(init)

    // 3) 异步刷新一次真实数据
    this.refreshOneForm(formId).catch(() => {})

    return binding
  }

  onUpdateForm(formId: string): void {
    this.refreshOneForm(formId).catch(() => {})
  }

  private async refreshOneForm(formId: string): Promise<void> {
    const bind: FormBindingEntry | null = await this.loadFormBinding(formId)
    const type: 'stock' | 'index' = bind ? bind.type : this.DEFAULT_TYPE
    const code: string = bind ? bind.code : this.DEFAULT_CODE

    const q: BackendQuote | null = await this.fetchBackendQuote(type, code)
    const data: WidgetBindData = this.buildWidgetData(formId, type, code, q)

    const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(data)
    await formProvider.updateForm(formId, binding)
  }

  private buildWidgetData(
    formId: string,
    type: 'stock' | 'index',
    code: string,
    q: BackendQuote | null
  ): WidgetBindData {
    const now: Date = new Date()
    const hh: string = now.getHours().toString().padStart(2, '0')
    const mm: string = now.getMinutes().toString().padStart(2, '0')

    if (q === null) {
      return {
        formId: formId,
        title: type === 'index' ? '指数' : '股票',
        code: code,
        name: '网络异常',
        priceText: '--',
        changeText: '--',
        pctText: '--',
        tsText: `${hh}:${mm}`
      }
    }

    return {
      formId: formId,
      title: type === 'index' ? '指数' : '股票',
      code: q.code,
      name: q.name,
      priceText: q.price.toFixed(2),
      changeText: q.change.toFixed(2),
      pctText: q.change_percent.toFixed(2) + '%',
      tsText: `${hh}:${mm}`
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

  // ======== 读取 formId（无 unknown）========
  private getFormIdFromWant(want: Want): string {
    if (!want.parameters) {
      return ''
    }

    // parameters 在 ArkTS 下可安全按 Object 处理
    const params: Record<string, Object> = want.parameters as Record<string, Object>

    // 方式 1：FormParam.IDENTITY_KEY
    const k1: string = formInfo.FormParam.IDENTITY_KEY
    const s1: string = this.readStringParam(params, k1)
    if (s1.length > 0) {
      return s1
    }

    // 方式 2：兼容字段
    const s2: string = this.readStringParam(params, 'ohos.extra.param.key.form_identity')
    if (s2.length > 0) {
      return s2
    }

    return ''
  }

  private readStringParam(params: Record<string, Object>, key: string): string {
    const v: Object | undefined = params[key]
    if (v === undefined || v === null) {
      return ''
    }
    // 这里不需要 unknown：typeof 在运行期判断真实类型即可
    if (typeof v === 'string') {
      const s: string = v as string
      return s.trim()
    }
    return ''
  }

  // ======== formId -> (type, code) 持久化（数组条目）========
  private async saveFormBinding(formId: string, type: 'stock' | 'index', code: string): Promise<void> {
    if (formId.length === 0) return
    if (code.length !== 6) return

    const pref = await preferences.getPreferences(this.context, this.PREF_FILE)
    const jsonStr: string = (await pref.get(this.PREF_KEY_FORM_ENTRIES, '')) as string

    const list: FormBindingEntry[] = this.safeParseEntries(jsonStr)

    let replaced: boolean = false
    for (let i: number = 0; i < list.length; i++) {
      if (list[i].formId === formId) {
        list[i] = { formId: formId, type: type, code: code }
        replaced = true
        break
      }
    }
    if (!replaced) {
      list.push({ formId: formId, type: type, code: code })
    }

    await pref.put(this.PREF_KEY_FORM_ENTRIES, JSON.stringify(list))
    await pref.flush()
  }

  private async loadFormBinding(formId: string): Promise<FormBindingEntry | null> {
    if (formId.length === 0) return null

    const pref = await preferences.getPreferences(this.context, this.PREF_FILE)
    const jsonStr: string = (await pref.get(this.PREF_KEY_FORM_ENTRIES, '')) as string

    const list: FormBindingEntry[] = this.safeParseEntries(jsonStr)
    for (let i: number = 0; i < list.length; i++) {
      if (list[i].formId === formId) {
        return list[i]
      }
    }
    return null
  }

  private safeParseEntries(jsonStr: string): FormBindingEntry[] {
    try {
      const raw: Object = JSON.parse(jsonStr) as Object
      if (!Array.isArray(raw)) {
        return []
      }

      const arr: Object[] = raw as Object[]
      const out: FormBindingEntry[] = []

      for (let i: number = 0; i < arr.length; i++) {
        const itObj: Object = arr[i]
        if (itObj === null || typeof itObj !== 'object' || Array.isArray(itObj)) {
          continue
        }

        const rec: Record<string, Object> = itObj as Record<string, Object>

        const formIdObj: Object | undefined = rec['formId']
        const typeObj: Object | undefined = rec['type']
        const codeObj: Object | undefined = rec['code']

        if (typeof formIdObj !== 'string' || typeof typeObj !== 'string' || typeof codeObj !== 'string') {
          continue
        }

        const formId: string = (formIdObj as string).trim()
        const typeStr: string = (typeObj as string).trim()
        const code: string = (codeObj as string).trim()

        if (formId.length === 0) continue
        if (code.length !== 6) continue
        if (typeStr !== 'stock' && typeStr !== 'index') continue

        const type: 'stock' | 'index' = (typeStr === 'stock') ? 'stock' : 'index'
        out.push({ formId: formId, type: type, code: code })
      }

      return out
    } catch (e) {
      return []
    }
  }
}
