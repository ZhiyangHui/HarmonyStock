import http from '@ohos.net.http'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { NotifyCenter, QuoteAlertPayload } from './NotifyCenter'

// 系统用的日志分组号
const DOMAIN: number = 0x0000
const TAG: string = 'HarmonyStock.Rule'

// 行情类型：用于区分指数和股票（影响接口参数和缓存 key）
export type QuoteType = 'index' | 'stock'

// 规则引擎内部使用的“轻量版通知规则”
// 只保留判断和触发通知所必需的字段，不掺 UI 状态
export interface NotifyRuleLite {
  id: number
  quoteType: QuoteType      // 规则作用对象：指数 / 股票
  stockCode: string         // 6 位代码
  stockName: string         // 展示用名称（发通知时用）
  conditionText: string     // 条件描述文本，例如“价格 ≥ 12.34”
  enabled: boolean          // 是否启用该规则
}

// 后端行情接口返回的原始结构
// 字段名保持与接口一致（如 change_percent）
interface BackendQuote {
  code: string              // 标的代码
  name: string              // 标的名称
  price: number             // 当前价格
  change: number            // 涨跌额
  change_percent: number    // 涨跌幅（百分比）
}

// 支持的比较运算符
// 只覆盖当前 UI 中允许用户选择的几种
type Op = '>' | '<' | '>=' | '<='

// 解析 conditionText 后得到的结构
// 用于后续直接做数值判断
interface ParsedRule {
  ok: boolean               // 是否成功解析
  op: Op                    // 比较符号
  threshold: number         // 阈值（数值）
}

function parseConditionText(t: string): ParsedRule {
  // 去掉中文前缀“价格”，并清理首尾空格
  // 例："价格 ≥ 10.5" -> "≥ 10.5"
  const cleaned: string = t.replace('价格', '').trim()

  // 统一运算符表示：
  //   中文 ≥ ≤  ->  >= <=
  //   同时移除所有空格，方便后续解析
  // 例："≥ 10.5" -> ">=10.5"
  const norm: string = cleaned
    .replace('≥', '>=')
    .replace('≤', '<=')
    .replace(/\s+/g, '')

  // 支持的比较运算符（顺序很重要，>= 要在 > 前面）
  const ops: Op[] = ['>=', '<=', '>', '<']

  // 遍历查找字符串中包含的运算符
  for (let i: number = 0; i < ops.length; i++) {
    const op: Op = ops[i]
    const pos: number = norm.indexOf(op)

    // 找到了某个运算符
    if (pos >= 0) {
      // 截取运算符后面的数值部分
      // 例：">=10.5" -> "10.5"
      const numStr: string = norm.substring(pos + op.length)

      // 转成 number
      const v: number = Number(numStr)

      // 数字合法，返回解析成功的规则
      if (!Number.isNaN(v)) {
        return { ok: true, op, threshold: v }
      }
    }
  }

  // 没能解析出合法规则，返回失败兜底
  return { ok: false, op: '>=', threshold: 0 }
}

function hit(op: Op, price: number, thr: number): boolean {
  if (op === '>=') return price >= thr
  if (op === '<=') return price <= thr
  if (op === '>') return price > thr
  return price < thr
}


export class RuleEngine {
  // RuleEngine 是否正在运行（防止重复 start）
  private static running: boolean = false

  // 定时轮询间隔（毫秒），用于周期性检查行情是否命中规则
  private static intervalMs: number = 15000

  // ===== 命中频控相关 =====

  // 同一条规则命中后，在 cooldownMs 时间内不再重复通知（防刷）
  private static cooldownMs: number = 60000

  // 记录每条规则上一次触发通知的时间
  // key: ruleId, value: last fire timestamp (ms)
  private static lastFireAt: Map<number, number> = new Map<number, number>()

  // ===== 规则来源 =====

  // 由外部注入的规则提供函数
  // RuleEngine 每次 tick 时都会调用它，获取“最新”的规则列表
  private static rulesProvider: (() => NotifyRuleLite[]) | null = null


  // ===== 启动行情规则引擎：定期拉取规则并执行命中检测 =====
  static start(
    provider: () => NotifyRuleLite[],   // 外部传入：返回当前最新规则列表的函数
    intervalMs: number,                 // 轮询间隔（毫秒）
    cooldownMs?: number                 // 可选：同一规则命中后的冷却时间
  ): void {
    // 记录一次启动日志，方便调试生命周期问题
    hilog.info(0x0000, 'HarmonyStock.Rule', 'RuleEngine.start called')

    // 防止重复启动（多次 start 只生效一次）
    if (RuleEngine.running) return

    // 标记引擎已运行
    RuleEngine.running = true

    // 保存规则提供函数，loop 中会周期性调用它
    RuleEngine.rulesProvider = provider

    // 设置轮询间隔，做下限保护（避免过于频繁）
    if (intervalMs > 1000) {
      RuleEngine.intervalMs = intervalMs
    }

    // 如果传入了 cooldownMs，则覆盖默认冷却时间
    if (cooldownMs !== undefined && cooldownMs >= 0) {
      RuleEngine.cooldownMs = cooldownMs
    }

    // 启动异步循环（不 await，避免阻塞调用方）
    // 内部是 while(running) + sleep 的长期任务
    RuleEngine.loop().catch(() => {})
  }


  static stop(): void {
    RuleEngine.running = false
  }

  // ===== 规则引擎主循环：周期性拉取规则并执行一次检查 =====
  private static async loop(): Promise<void> {
    // 如果引擎已被停止，直接退出，不再继续循环
    if (!RuleEngine.running) return

    try {
      // 取出外部注入的规则提供函数
      const provider: (() => NotifyRuleLite[]) | null = RuleEngine.rulesProvider

      // 通过 provider 获取“当前最新”的规则快照
      // 注意：这里是同步取规则，规则本身通常来自内存状态
      const rules: NotifyRuleLite[] = provider ? provider() : []

      // 对这一批规则执行一次完整的命中检查
      await RuleEngine.checkOnce(rules)

    } catch (e) {
      // 单次 loop 出错不影响后续轮询，只记录日志
      hilog.error(DOMAIN, TAG, 'loop err=%{public}s', JSON.stringify(e))
    }

    // 使用 setTimeout 递归调用自身，形成“手写定时循环”
    // 不保存返回的 timerId，避免引入 any / unknown
    setTimeout(() => {
      RuleEngine.loop().catch(() => {})
    }, RuleEngine.intervalMs)
  }


  // ===== 单轮规则检查：逐条拉价格 -> 判断条件 -> 命中就发通知（带冷却） =====
  private static async checkOnce(rules: NotifyRuleLite[]): Promise<void> {
    // 顺序遍历规则列表：一条一条处理，避免并发把后端打爆、也避免通知乱序
    for (let i: number = 0; i < rules.length; i++) {
      const r: NotifyRuleLite = rules[i]

      // 规则被关闭就直接跳过
      if (!r.enabled) continue

      // 把“价格 ≥ 220”这种文案解析成 {op, threshold}
      const parsed: ParsedRule = parseConditionText(r.conditionText)
      // 解析失败的规则不参与检查（通常是文案格式不对）
      if (!parsed.ok) continue

      // 按规则类型去拉当前价：stock / index 走同一个接口，只是 type 参数不同
      const price: number | null = await RuleEngine.fetchPrice(r.quoteType, r.stockCode)

      // 这行日志建议放到“命中后”再打；你现在放这里会导致每次轮询都刷 “rule hit”
      hilog.info(DOMAIN, TAG, 'rule hit id=%{public}d type=%{public}s code=%{public}s', r.id, r.quoteType, r.stockCode)

      // 行情没拉到（网络/404/解析失败）就跳过本条规则
      if (price === null) continue

      // 条件不满足就跳过（例如当前价没到阈值）
      if (!hit(parsed.op, price, parsed.threshold)) continue

      // ===== 频控：同一条规则命中后，cooldownMs 内不重复通知 =====
      const now: number = Date.now()
      const last: number | undefined = RuleEngine.lastFireAt.get(r.id)
      if (last !== undefined && now - last < RuleEngine.cooldownMs) {
        continue
      }
      // 记录本次触发时间，下一次命中会走冷却判断
      RuleEngine.lastFireAt.set(r.id, now)

      // 组装系统通知的 payload：id 做偏移，避免和别的通知 id 撞
      const p: QuoteAlertPayload = {
        id: 10000 + r.id,
        title: '行情提醒',
        text: `${r.stockName}（${r.stockCode}）${r.conditionText}，当前价 ${price.toFixed(2)}`,
        additionalText: '打开应用查看详情',
        bundleName: 'com.example.harmonystock',
        abilityName: 'EntryAbility'
      }

      // 触发系统通知（内部会确保通知权限开启）
      await NotifyCenter.publish(p)

      // 这条日志才是真正意义上的“命中并已通知”
      hilog.info(DOMAIN, TAG, 'rule hit id=%{public}d code=%{public}s', r.id, r.stockCode)
    }
  }


  // 从后端按类型（股票 / 指数）拉取当前价格，仅用于通知规则判断
  private static async fetchPrice(
    quoteType: QuoteType,
    code6: string
  ): Promise<number | null> {

    // 基本校验：只接受 6 位代码
    const code: string = code6.trim()
    if (code.length !== 6) return null

    const httpRequest = http.createHttp()
    try {
      // 关键点：这里的 type 直接使用规则里的 quoteType，
      // 避免股票 / 指数同码（如 000001）查错行情
      const url: string =
        `http://101.43.185.73:8000/api/quote/by_code?type=${quoteType}&code=${code}`

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        // 通知轮询场景下，允许稍长超时，避免弱网误判
        connectTimeout: 55000,
        readTimeout: 55000
      })

      // 请求成功且有返回内容时，解析价格
      if (res.responseCode === 200 && res.result) {
        const raw: BackendQuote = JSON.parse(res.result as string) as BackendQuote

        hilog.info(
          DOMAIN,
          TAG,
          'fetchPrice ok type=%{public}s code=%{public}s price=%{public}s',
          quoteType,
          code,
          raw.price.toFixed(2)
        )

        return raw.price
      }

      // 非 200 或无数据，统一视为失败
      return null
    } catch (e) {
      // 网络 / 解析异常直接吞掉，避免打断整个规则循环
      hilog.error(DOMAIN, TAG, 'fetchPrice err=%{public}s', JSON.stringify(e))
      return null
    } finally {
      // 主动释放 http 实例，避免资源泄漏
      httpRequest.destroy()
    }
  }

}
