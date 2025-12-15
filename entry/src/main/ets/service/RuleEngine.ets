import http from '@ohos.net.http'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { NotifyCenter, QuoteAlertPayload } from './NotifyCenter'

const DOMAIN: number = 0x0000
const TAG: string = 'HarmonyStock.Rule'

export interface NotifyRuleLite {
  id: number
  stockCode: string
  stockName: string
  conditionText: string
  enabled: boolean
}

interface BackendQuote {
  code: string
  name: string
  price: number
  change: number
  change_percent: number
}

type Op = '>' | '<' | '>=' | '<='

interface ParsedRule {
  ok: boolean
  op: Op
  threshold: number
}

/**
 * 只解析你目前的 conditionText：
 *  - "价格 ≥ 220.00"
 *  - "价格 <= 28.00"
 *  - ">=220" 也能兜底
 */
function parseConditionText(t: string): ParsedRule {
  // 去掉“价格”等中文前缀、空格
  const cleaned: string = t.replace('价格', '').trim()

  // 统一符号：≥ ≤ -> >= <=
  const norm: string = cleaned
    .replace('≥', '>=')
    .replace('≤', '<=')
    .replace(/\s+/g, '')

  const ops: Op[] = ['>=', '<=', '>', '<']
  for (let i: number = 0; i < ops.length; i++) {
    const op: Op = ops[i]
    const pos: number = norm.indexOf(op)
    if (pos >= 0) {
      const numStr: string = norm.substring(pos + op.length)
      const v: number = Number(numStr)
      if (!Number.isNaN(v)) {
        return { ok: true, op, threshold: v }
      }
    }
  }

  return { ok: false, op: '>=', threshold: 0 }
}

function hit(op: Op, price: number, thr: number): boolean {
  if (op === '>=') return price >= thr
  if (op === '<=') return price <= thr
  if (op === '>') return price > thr
  return price < thr
}

/**
 * RuleEngine：轮询行情、判断规则、触发系统通知
 * - 不使用 any / unknown
 * - 不保存 setInterval / setTimeout 的返回值，避免类型不一致
 */
export class RuleEngine {
  private static running: boolean = false
  private static intervalMs: number = 15000

  // 频控：同一条规则命中后 cooldownMs 内不重复通知
  private static cooldownMs: number = 60000
  private static lastFireAt: Map<number, number> = new Map<number, number>()

  // 外部提供最新规则（每次 tick 都取一次）
  private static rulesProvider: (() => NotifyRuleLite[]) | null = null

  static start(provider: () => NotifyRuleLite[], intervalMs: number, cooldownMs?: number): void {
    hilog.info(0x0000, 'HarmonyStock.Rule', 'RuleEngine.start called')
    if (RuleEngine.running) return
    RuleEngine.running = true
    RuleEngine.rulesProvider = provider

    if (intervalMs > 1000) {
      RuleEngine.intervalMs = intervalMs
    }
    if (cooldownMs !== undefined && cooldownMs >= 0) {
      RuleEngine.cooldownMs = cooldownMs
    }

    RuleEngine.loop().catch(() => {})
  }

  static stop(): void {
    RuleEngine.running = false
    // 不清 timerId（我们没有存），下一个 tick 会自然退出
  }

  private static async loop(): Promise<void> {
    if (!RuleEngine.running) return

    try {
      const provider: (() => NotifyRuleLite[]) | null = RuleEngine.rulesProvider
      const rules: NotifyRuleLite[] = provider ? provider() : []
      await RuleEngine.checkOnce(rules)
      
    } catch (e) {
      hilog.error(DOMAIN, TAG, 'loop err=%{public}s', JSON.stringify(e))
    }

    // 递归 setTimeout：不保存返回值，避免 any/unknown
    setTimeout(() => {
      RuleEngine.loop().catch(() => {})
    }, RuleEngine.intervalMs)
  }

  private static async checkOnce(rules: NotifyRuleLite[]): Promise<void> {
    for (let i: number = 0; i < rules.length; i++) {
      const r: NotifyRuleLite = rules[i]
      if (!r.enabled) continue

      const parsed: ParsedRule = parseConditionText(r.conditionText)
      if (!parsed.ok) continue

      const price: number | null = await RuleEngine.fetchPrice(r.stockCode)
      if (price === null) continue

      if (!hit(parsed.op, price, parsed.threshold)) continue

      // 频控
      const now: number = Date.now()
      const last: number | undefined = RuleEngine.lastFireAt.get(r.id)
      if (last !== undefined && now - last < RuleEngine.cooldownMs) {
        continue
      }
      RuleEngine.lastFireAt.set(r.id, now)

      const p: QuoteAlertPayload = {
        id: 10000 + r.id,
        title: '行情提醒',
        text: `${r.stockName}（${r.stockCode}）${r.conditionText}，当前价 ${price.toFixed(2)}`,
        additionalText: '打开应用查看详情'
      }

      await NotifyCenter.publish(p)
      hilog.info(DOMAIN, TAG, 'rule hit id=%{public}d code=%{public}s', r.id, r.stockCode)
    }
  }

  private static async fetchPrice(code6: string): Promise<number | null> {
    const code: string = code6.trim()
    if (code.length !== 6) return null

    const httpRequest = http.createHttp()
    try {
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=stock&code=${code}`
      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 55000,
        readTimeout: 55000
      })

      if (res.responseCode === 200 && res.result) {
        const raw: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        hilog.info(
          DOMAIN,
          TAG,
          'fetchPrice ok code=%{public}s price=%{public}s',
          code,
          raw.price.toFixed(2)
        )
        return raw.price
      }
      return null
    } catch (e) {
      hilog.error(DOMAIN, TAG, 'fetchPrice err=%{public}s', JSON.stringify(e))
      return null
    } finally {
      httpRequest.destroy()
    }
  }
}
