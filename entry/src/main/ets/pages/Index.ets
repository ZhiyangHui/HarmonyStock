import http from '@ohos.net.http';
import preferences from '@ohos.data.preferences';
import { formBindingData, formProvider } from '@kit.FormKit';
import { AppStorageV2 } from '@kit.ArkUI'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { ActiveFormState, APP_STORAGE_KEY_ACTIVE_FORM } from '../common/ActiveFormatState'

import notificationManager from '@ohos.notificationManager'
import notification from '@ohos.notification'
import { RuleEngine, NotifyRuleLite } from '../service/RuleEngine'
import wantAgent from '@ohos.wantAgent'

const DOMAIN = 0x0000
const TAG: string = 'HarmonyStock'


// 前端统一的行情对象（指数/股票共用，UI/缓存/K线都用它）
interface Quote {
  code: string
  name: string
  price: number
  change: number
  changePercent: number
  type: 'index' | 'stock' // 用来区分指数/股票
}

// 后端原始返回结构（字段名按后端来，用 mapBackendQuote 转成 Quote）
interface BackendQuote {
  code: string
  name: string
  price: number
  change: number
  change_percent: number // 后端用下划线命名；前端映射到 changePercent
}

// 自选页展示用的简化行情结构（不一定每处都要用，但保留方便）
interface StockQuote {
  name: string
  code: string
  price: number
  change: number
  changePercent: number
}

// 通知规则里用到的标的类型（决定 RuleEngine 请求哪个 type）
type QuoteType = 'index' | 'stock'

// 通知规则（存本地 + RuleEngine 轮询判断命中）
interface NotifyRule {
  id: number
  quoteType: QuoteType      // 规则针对指数还是股票（不加这个会有同码冲突问题）
  stockCode: string         // 6位代码（
  stockName: string
  conditionText: string     // 例如 "价格 ≥ 25.00"（RuleEngine 会解析这个字符串）
  enabled: boolean
}

// K线点（后端返回的OHLCV）
interface KLinePoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// K线渲染用的几何信息（把“价格”预处理成“高度/偏移”，UI直接画Rect）
interface KlineGeom {
  upperWick: number
  bodyHeight: number
  lowerWick: number
  topGap: number
  isUp: boolean // close >= open
}

// 自选里存的“股票引用”（只存code，name可选）
interface StockRef {
  code: string
  name?: string
}

// 一个自选分组（stocks变化时 version +1，用来强制List刷新）
interface StockGroup {
  id: number
  name: string
  stocks: StockRef[]
  createdAt: number
  version: number // 用于触发 UI 识别结构变化（尤其是 List/ForEach 场景）
}

// 分组统计（数值版，适合计算）
interface GroupStats {
  groupId: number
  stockCount: number
  upCount: number
  downCount: number
  avgChangePercent: number
}

// 分组统计（展示版，直接给Text用；缺数据时用 '--'）
interface ActiveGroupStats {
  stockCount: string
  up: string
  down: string
  avgPct: string // 例如 "0.52%" 或 "--"
}

// 桌面卡片绑定数据（updateForm用；多数是已经格式化好的字符串）
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


// 页面组件
@Entry
@Component
struct HarmonyStockPage {
  // 顶部 Tab：0 市场 / 1 自选 / 2 通知（Tabs.onChange 会改它）
  @State currentTabIndex: number = 0

  // 当前行情模式：指数 or 股票（影响行情列表、搜索、K线请求 type）
  @State currentMode: 'index' | 'stock' = 'index'

  // 行情横向 strip 的加载/报错状态（只管“市场页”的那一条）
  @State listLoading: boolean = false
  @State listError: string = ''

  // 搜索输入与状态（输入框 + 查询按钮那块）
  @State searchCode: string = ''          // 输入的 6 位代码
  @State searchLoading: boolean = false
  @State searchError: string = ''

  // 行情卡片数据：实时列表 + 搜索结果（都显示在同一个横向 strip 里）
  @State marketQuotes: Quote[] = []       // 后端批量返回的实时卡片
  @State searchedQuotes: Quote[] = []     // 查询出来的卡片（置顶、去重）

  // 当前选中的标的（决定 K 线展示的是谁，也用于桌面卡片回写）
  @State selectedQuoteCode: string = ''   // 6 位代码
  @State selectedQuoteName: string = ''   // UI 标题用
  @State selectedQuoteType: 'index' | 'stock' = 'index' // K线请求 type / 缓存key用

  // 当前 K 线周期：0 日K / 1 周K / 2 月K（切换按钮用）
  @State klinePeriodIndex: number = 0

  // K 线请求状态（加载/错误/数据）
  @State klineLoading: boolean = false
  @State klineError: string = ''
  @State klinePoints: KLinePoint[] = []  // 后端返回的 OHLCV 点

  // K 线渲染用的“预计算”结果（把点算成高度/偏移，UI直接画）
  @State klineGeoms: KlineGeom[] = []
  @State klineMinPrice: number = 0       // y轴下边界（显示刻度用）
  @State klineMaxPrice: number = 0       // y轴上边界（显示刻度用）
  @State klineXLabels: string[] = []     // x轴标签（最多取几段）

  // K 线布局常量（计算和 UI 高度必须一致，否则会溢出/压缩）
  private readonly KLINE_PLOT_H: number = 180     // 蜡烛绘图区高度
  private readonly KLINE_LABEL_H: number = 24     // x 轴标签区域高度
  // private readonly KLINE_CARD_INNER_PAD: number = 16 // K线卡片内部 padding（留着统一改）

  // 当前选中的那根 K 线（点某根蜡烛后，详情条显示这些值）
  @State selectedKlineIndex: number = -1
  @State selectedKlineDate: string = ''
  @State selectedKlineOpen: number = 0
  @State selectedKlineHigh: number = 0
  @State selectedKlineLow: number = 0
  @State selectedKlineClose: number = 0

  // 蜡烛宽度与间距（影响总宽度与滚动手感）
  private candleBodyWidth: number = 8
  private candleGap: number = 4

  // expandedKeys 统一控制展开项（现在可不用）
  @State expandedKeys: string[] = []

  // 记录每张行情卡片是否展开：key = `${type}-${code}`（解决同码/不同类型的冲突）
  @State cardExpandedMap: Map<string, boolean> = new Map<string, boolean>()

  // 网络请求超时（行情/K线/单只查询都用这套）
  private readonly HTTP_CONNECT_TIMEOUT: number = 55000
  private readonly HTTP_READ_TIMEOUT: number = 55000

  // 自选分组本地存储（preferences 文件名 + key）
  private readonly PREF_FILE: string = 'harmonystock_watchlist'
  private readonly PREF_KEY_GROUPS: string = 'groups_json'
  private nextGroupId: number = 1         // 新建分组用的自增 id

  // 自选页当前分组与输入状态
  @State activeGroupId: number = -1       // 当前选中的分组 id
  @State watchlistMsg: string = ''        // 自选页/操作反馈的提示文案
  @State newGroupName: string = ''        // 新建分组输入框
  @State addStockCodeInput: string = ''   // 添加股票输入框

  // 行情缓存：key = `${type}-${code}`（自选统计/名称/涨跌依赖它）
  @State quoteCache: Map<string, Quote> = new Map<string, Quote>()
  @State quoteCacheVer: number = 0        // 强制触发依赖缓存的 UI 重新计算

  // 当前 active 分组的股票列表（从 groups 派生出来，单独存一份方便 List 渲染）
  @State activeStocks: StockRef[] = []
  @State activeStocksVer: number = 0      // 强制 List 重新创建/刷新用

  // 自选统计卡上的展示值（这里直接存 string，避免 UI 里到处 toFixed/拼%）
  @State activeStatStockCount: string = '0'
  @State activeStatUp: string = '0'
  @State activeStatDown: string = '0'
  @State activeStatAvgPct: string = '--'
  @State activeStatsVer: number = 0       // 配合 key，用来让统计卡稳定刷新

  // 从桌面卡片拉起 App 时的 formId（用于把“当前选择”回写到桌面卡片）
  @State activeFormId: string = ''

  // 通知页：新增规则输入区（股票代码/操作符/阈值/提示）
  @State ruleStockCodeInput: string = ''  // 规则代码输入
  @State ruleOpIndex: number = 0          // 0: ≥  1: ≤（你现在只做这俩就够稳）
  @State rulePriceInput: string = ''      // 价格阈值输入
  @State ruleMsg: string = ''             // 新增规则时的提示/校验信息
  @State ruleQuoteType: QuoteType = 'stock' // 规则针对指数还是股票（默认股票更符合习惯）

  // 通知规则列表版本号（每次增删/切换 enabled 都 ++，用来刷新 List）
  @State notifyRulesVer: number = 0

  onPageShow() {
    // 页面回到前台时触发：先拿到桌面卡片传来的 formId，再（如果有选中的标的）顺手把当前行情回写到卡片
    hilog.error(0x0000, 'INDEX_DEBUG', '🔥 Index onPageShow CALLED')

    // 从 AppStorageV2 消费 formId（用于后续更新桌面卡片）
    this.consumeFormIdFromAppStorageV2('onPageShow')

    // 如果是从卡片打开的，并且当前已经选中了一只标的：同步更新卡片展示
    if (this.activeFormId.length > 0 && this.selectedQuoteCode.length > 0) {
      const q: Quote | null = this.findQuoteInCaches(this.selectedQuoteCode)
      if (q !== null) {
        this.updateWidgetIfLaunchedFromForm(q)
      }
    }
  }

  // ===== 自选 / 通知假数据（保持原样） =====
  @State groups: StockGroup[] = [
    {
      id: 1,
      name: '科技股',
      stocks: [
        { name: '宁德时代', code: '300750' },
        { name: '中芯国际', code: '688981' }
      ],
      createdAt: Date.now(),
      version: 0
    },
    {
      id: 2,
      name: '新能源',
      stocks: [
        { name: '隆基绿能', code: '601012' }
      ],
      createdAt: Date.now(),
      version: 0
    }
  ];

  // 通知规则列表：UI 里展示、RuleEngine 轮询时也从这里取（会持久化到 prefs）
  @State notifyRules: NotifyRule[] = [
    // 示例规则：宁德时代，股票类型；当价格达到/超过阈值时触发通知
    { id: 999, quoteType: 'stock', stockCode: '300750', stockName: '宁德时代', conditionText: '价格 ≥ 1.00', enabled: true },

    // 示例规则：隆基绿能，股票类型；当价格低于/等于阈值时触发通知（这里先关掉）
    { id: 2, quoteType: 'stock', stockCode: '601012', stockName: '隆基绿能', conditionText: '价格 ≤ 28.00', enabled: true }
  ]

  // 下一条规则的自增 id 起点：从 prefs 读完规则后会用 recomputeNextRuleId() 重新校准，避免重复
  private nextRuleId: number = 1000

  // 通知规则在 preferences 里保存用的 key（对应 PREF_FILE 里的一个字段名）
  private readonly PREF_KEY_RULES: string = 'notify_rules_json'

  private refreshActiveGroupStats(): void {
    // 重新根据当前分组的股票列表（activeStocks）计算统计数据：数量/涨跌/均涨跌
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.activeStocks)

    // 写回到几个 @State 字段，UI 里统计卡片会直接绑定这些值
    this.activeStatStockCount = st.stockCount
    this.activeStatUp = st.up
    this.activeStatDown = st.down
    this.activeStatAvgPct = st.avgPct

    // 版本号手动 +1：有些地方用 key 依赖它，确保统计区域能稳定刷新
    this.activeStatsVer += 1

    // 调试用：看一下每次刷新后的统计结果
    console.info(
      `[DBG] stats=${this.activeStatStockCount}, up=${this.activeStatUp}, down=${this.activeStatDown}, avg=${this.activeStatAvgPct}, ver=${this.activeStatsVer}`
    )
  }


  // 后端行情 -> 前端 Quote
  private mapBackendQuote(raw: BackendQuote, type: 'index' | 'stock'): Quote {
    return {
      code: raw.code,
      name: raw.name,
      price: raw.price,
      change: raw.change,
      changePercent: raw.change_percent,
      type: type
    };
  }

  // 页面生命周期
  aboutToAppear() {
    // 页面即将显示：打个调试点，方便看生命周期是否正常走到这
    this.dbg('aboutToAppear:enter')

    // 先把本地持久化的数据恢复出来（分组/规则/自选预热/拉行情）
    // 这里用 then 是为了确保顺序：分组 -> 规则 -> 预热 -> 拉列表
    this.loadGroupsFromPrefs().then(async () => {
      await this.loadNotifyRulesFromPrefs()   // 读取本地保存的通知规则（notify_rules_json）
      await this.preloadActiveGroupQuotes()   // 预热当前分组的股票行情到缓存，后面统计/列表能立刻显示
      this.fetchQuotes(this.currentMode)      // 拉取当前模式（指数/股票）的市场列表
    })

    // 启动规则引擎：内部会定时轮询行情并判断规则是否命中
    // 这里传的是一个 provider：每次 tick 都从页面最新的 notifyRules 里取一份轻量规则
    RuleEngine.start((): NotifyRuleLite[] => {
      const out: NotifyRuleLite[] = []

      for (let i: number = 0; i < this.notifyRules.length; i++) {
        const r: NotifyRule = this.notifyRules[i]

        // 注意：RuleEngine 只认识 NotifyRuleLite，所以要把页面规则“映射”过去
        // quoteType 必须带上（股票/指数），不然会出现你之前的缺字段报错
        out.push({
          id: r.id,
          quoteType: r.quoteType,      // stock / index（决定后端拉价时 type=stock 还是 type=index）
          stockCode: r.stockCode,
          stockName: r.stockName,
          conditionText: r.conditionText,
          enabled: r.enabled
        })
      }
      return out
    }, 5000, 10000) // intervalMs=5000：每 5s 检查一次；cooldownMs=10000：命中后 10s 内不重复通知
  }

  aboutToDisappear(): void {
    // 页面离开时停止轮询，避免后台一直跑（省电/省流量，也避免重复通知）
    RuleEngine.stop()
  }


  // 拉取行情列表（指数 / 股票共用）
  // type 传 'index' 或 'stock'，同一套接口走两种数据
  private async fetchQuotes(type: 'index' | 'stock'): Promise<void> {
    // 只有“请求的类型”刚好是当前页面模式时，才更新 loading / error / 列表 UI
    // 这样你以后如果后台偷偷拉另一种类型的数据，也不会把用户正在看的界面刷乱
    const affectUI: boolean = (type === this.currentMode)

    if (affectUI) {
      this.listLoading = true     // 页面上的“行情加载中…”状态
      this.listError = ''         // 清空上一次错误
    }

    const httpRequest = http.createHttp()
    try {
      // 后端根据 type 返回指数列表或股票列表
      const url: string = `http://101.43.185.73:8000/api/quote?type=${type}`

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      if (res.responseCode === 200 && res.result) {
        // 后端字段是 change_percent，这里统一映射成前端 Quote 结构
        const raw: BackendQuote[] = JSON.parse(res.result as string) as BackendQuote[]
        const mapped: Quote[] = raw.map((it: BackendQuote): Quote => this.mapBackendQuote(it, type))

        // 只在“当前模式”下才更新市场卡片列表
        if (affectUI) {
          this.marketQuotes = mapped
        }

        // 不管当前 UI 在看哪种模式，都把数据写进缓存：
        // 自选分组统计 / 自选列表行显示 都依赖 quoteCache
        this.mergeQuotesToCache(mapped)

        // 第一次进入页面：默认选中第一条，并拉一份 K 线
        // 只对当前模式生效，避免切到另一模式时误触发
        if (affectUI && this.selectedQuoteCode === '' && mapped.length > 0) {
          const first: Quote = mapped[0]
          this.selectedQuoteCode = first.code
          this.selectedQuoteName = first.name
          this.selectedQuoteType = first.type

          const period: string = this.getKlinePeriodParam()
          this.fetchKline(first.type, first.code, period) // 这里不 await，K 线自己走 loading
        }
      } else {
        // HTTP 非 200：只在当前模式下提示错误，避免“后台请求失败”影响用户正在看的界面
        if (affectUI) {
          this.listError = `HTTP ${res.responseCode}`
          this.marketQuotes = []
        }
      }
    } catch (e) {
      // 网络异常/解析异常：同样只影响当前模式 UI
      if (affectUI) {
        this.listError = `${e}`
        this.marketQuotes = []
      }
    } finally {
      // 释放 request 对象，避免资源泄漏
      httpRequest.destroy()

      // 只在当前模式下关闭 loading
      if (affectUI) {
        this.listLoading = false
      }
    }
  }

  // 拉取 K 线数据（指数 / 股票共用）
  // type: 'index' / 'stock'，code: 6位代码，period: day/week/month
  private async fetchKline(
    type: 'index' | 'stock',
    code: string,
    period: string
  ): Promise<void> {
    // 进入加载态：K 线卡片区域会显示“加载中…”
    this.klineLoading = true
    this.klineError = '' // 清空上一次错误

    const httpRequest = http.createHttp()
    try {
      // limit=60：最多取最近 60 根K线，前端蜡烛图宽度也按这个量来算
      const url: string =
        `http://101.43.185.73:8000/api/kline` +
          `?type=${type}&code=${code}&period=${period}&limit=60`

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      if (res.responseCode === 200 && res.result) {
        // 后端返回的就是 KLinePoint 数组（date/open/high/low/close/volume）
        const raw: KLinePoint[] = JSON.parse(res.result as string) as KLinePoint[]
        this.klinePoints = raw

        // 体验上一般默认看“最新一根”，所以这里选最后一个点
        // 同时会把详细信息条（开高低收）更新到对应的 @State 字段里
        if (this.klinePoints.length > 0) {
          const lastIndex: number = this.klinePoints.length - 1
          this.onKlineSelected(lastIndex)
        } else {
          // 没数据就清掉选中态，避免还显示旧的高亮/详情
          this.selectedKlineIndex = -1
        }

        // points 更新后，重新计算每根蜡烛的几何信息（上下影线/实体高度/顶部空白）
        // 这一步会同步更新：klineGeoms / klineMinPrice / klineMaxPrice / klineXLabels
        this.recomputeKlineGeom()
      } else {
        // HTTP 非 200：记录错误，并把图表相关数据清空（避免残留旧图）
        this.klineError = `HTTP ${res.responseCode}`
        this.klinePoints = []
        this.klineGeoms = []
        this.klineXLabels = []
        this.selectedKlineIndex = -1
      }
    } catch (e) {
      // 网络异常 / JSON 解析异常：同样清空图表数据，给 UI 一个可预期的状态
      this.klineError = `${e}`
      this.klinePoints = []
      this.klineGeoms = []
      this.klineXLabels = []
      this.selectedKlineIndex = -1
    } finally {
      // 释放 request 对象
      httpRequest.destroy()

      // 退出加载态
      this.klineLoading = false
    }
  }


  // 根据当前的周期选择（klinePeriodIndex）拼出后端需要的 period 参数
  // 约定：0=日K，1=周K，2=月K（和 UI 上的“日K/周K/月K”按钮对应）
  private getKlinePeriodParam(): string {
    if (this.klinePeriodIndex === 1) {
      return 'week'   // 周K：用于请求 ?period=week
    }
    if (this.klinePeriodIndex === 2) {
      return 'month'  // 月K：用于请求 ?period=month
    }
    return 'day'      // 默认日K：用于请求 ?period=day
  }

  // ===== 计算 K 线几何信息 & 坐标轴 =====
  private recomputeKlineGeom(): void {
    // points：后端返回的原始 K 线数据（开高低收 + 日期）
    const points: KLinePoint[] = this.klinePoints;
    // geoms：把“价格”换算成“像素高度”，UI 渲染蜡烛图只用它
    const geoms: KlineGeom[] = [];
    // labels：x 轴显示的日期（做了抽样，不会每根都画）
    const labels: string[] = [];

    // 没数据就清空 UI 对应状态
    if (points.length === 0) {
      this.klineGeoms = geoms;
      this.klineXLabels = labels;
      return;
    }

    // 1) 扫一遍，拿到这一屏 K 线的最高价 / 最低价（用于做 y 轴映射）
    let minPrice: number = points[0].low;
    let maxPrice: number = points[0].high;

    for (let i: number = 1; i < points.length; i++) {
      const p: KLinePoint = points[i];
      if (p.low < minPrice) {
        minPrice = p.low;
      }
      if (p.high > maxPrice) {
        maxPrice = p.high;
      }
    }

    // 极端情况防御：如果最高价=最低价，range=0，会导致除零
    let range: number = maxPrice - minPrice;
    if (range <= 0) {
      range = 1.0;
      maxPrice = minPrice + range;
    }

    // 记录给 y 轴刻度显示用
    this.klineMinPrice = minPrice;
    this.klineMaxPrice = maxPrice;

    // 2) 统一使用固定绘图高度（这必须和 UI 蜡烛区域 height 保持一致）
    const fullHeight: number = this.KLINE_PLOT_H;

    // 太扁的蜡烛看不见：给实体/影线一个最小像素高度
    const minBody: number = 4;
    const minWick: number = 2;

    // 把“价格”映射到“y 像素坐标”
    // y=0 在最上面；价格越高，y 越小
    const priceToY = (price: number): number => {
      const y: number = (maxPrice - price) / range * fullHeight;
      if (y < 0) {
        return 0;
      }
      if (y > fullHeight) {
        return fullHeight;
      }
      return y;
    };

    // 3) 把每根 K 线转成 UI 可直接画的几何数据
    for (let i: number = 0; i < points.length; i++) {
      const pt: KLinePoint = points[i];

      // 把开高低收都映射成 y 坐标（后面用这些来算实体/影线高度）
      const yHigh: number = priceToY(pt.high);
      const yLow: number = priceToY(pt.low);
      const yOpen: number = priceToY(pt.open);
      const yClose: number = priceToY(pt.close);

      // 实体的上边/下边：open 和 close 谁高谁在上
      let yTopBody: number = Math.min(yOpen, yClose);
      let yBottomBody: number = Math.max(yOpen, yClose);

      // 实体太薄时强制拉到最小高度（否则用户点选时像“没东西”）
      if (yBottomBody - yTopBody < minBody) {
        const mid: number = (yTopBody + yBottomBody) / 2.0;
        yTopBody = mid - minBody / 2.0;
        yBottomBody = mid + minBody / 2.0;

        // 拉伸后可能超出绘图区，做一次裁剪
        if (yTopBody < 0) {
          yTopBody = 0;
          yBottomBody = minBody;
        }
        if (yBottomBody > fullHeight) {
          yBottomBody = fullHeight;
          yTopBody = fullHeight - minBody;
        }
      }

      // 上影线高度：high -> 实体上边（y 越小越靠上，所以用 yTopBody - yHigh）
      let upperWick: number = yTopBody - yHigh;
      if (upperWick < minWick) {
        upperWick = minWick;
      }

      // 下影线高度：实体下边 -> low
      let lowerWick: number = yLow - yBottomBody;
      if (lowerWick < minWick) {
        lowerWick = minWick;
      }

      // topGap：蜡烛整体离顶部的空白（用于先画一个 Blank 把蜡烛“顶”到正确位置）
      let topGap: number = yHigh;
      if (topGap < 0) {
        topGap = 0;
      }
      if (topGap > fullHeight) {
        topGap = fullHeight;
      }

      // 再做一层防御：如果空白+影线+实体加起来超过绘图区，会导致布局溢出
      // 这里优先压缩 topGap（视觉影响最小），影线/实体尽量不动
      const totalH: number = topGap + upperWick + (yBottomBody - yTopBody) + lowerWick;
      if (totalH > fullHeight) {
        const overflow: number = totalH - fullHeight;
        topGap = topGap - overflow;
        if (topGap < 0) {
          topGap = 0;
        }
      }

      // 组装给 UI 用的结构：后面 ForEach 直接画 Rect
      geoms.push({
        upperWick: upperWick,
        bodyHeight: (yBottomBody - yTopBody),
        lowerWick: lowerWick,
        topGap: topGap,
        isUp: pt.close >= pt.open // 收>=开 视为涨（决定实体颜色）
      });
    }

    // 4) x 轴标签：最多显示 5 个，均匀抽样（不然 60 根 K 线会挤成一团）
    const total: number = points.length;
    const maxLabels: number = 5;
    const actualLabels: number = total >= maxLabels ? maxLabels : total;

    if (actualLabels > 0) {
      const lastIndex: number = total - 1;
      for (let i: number = 0; i < actualLabels; i++) {
        let idx: number;
        if (actualLabels === 1) {
          idx = 0;
        } else {
          idx = Math.floor(i * lastIndex / (actualLabels - 1));
        }

        // 日期一般是 YYYY-MM-DD，这里只显示 MM-DD 更省空间
        const dateStr: string = points[idx].date;
        const label: string = dateStr.length >= 5 ? dateStr.substring(5) : dateStr;
        labels.push(label);
      }
    }

    // 一次性写回状态，让 UI 刷新
    this.klineGeoms = geoms;
    this.klineXLabels = labels;
  }


  // 选中某一根 K 线
  private onKlineSelected(index: number): void {
    if (index < 0 || index >= this.klinePoints.length) {
      return;
    }

    const pt: KLinePoint = this.klinePoints[index];

    this.selectedKlineIndex = index;
    this.selectedKlineDate = pt.date;
    this.selectedKlineOpen = pt.open;
    this.selectedKlineHigh = pt.high;
    this.selectedKlineLow = pt.low;
    this.selectedKlineClose = pt.close;
  }

  // ===== 去掉 strip 中某个标的 =====
  private removeQuoteCard(code: string, type: 'index' | 'stock'): void {
    // 搜索结果里把对应的卡片删掉（同 code + 同 type 才算同一个）
    this.searchedQuotes = this.searchedQuotes.filter((item: Quote) =>
    !(item.code === code && item.type === type)
    );

    // 实时行情列表里也删掉一份
    this.marketQuotes = this.marketQuotes.filter((item: Quote) =>
    !(item.code === code && item.type === type)
    );

    // 如果删掉的是当前正在看的那只，就把右侧 K 线区域也一并清空，避免还显示旧数据
    if (this.selectedQuoteCode === code && this.selectedQuoteType === type) {
      this.selectedQuoteCode = '';
      this.selectedQuoteName = '';
      this.selectedKlineIndex = -1;
      this.klinePoints = [];
      this.klineGeoms = [];
      this.klineXLabels = [];
    }
  }


  // ===== 根据代码查询单个标的（指数 / 股票通用） =====
  private async queryQuoteByCode(): Promise<void> {
    // 输入框里拿到的代码先做一次 trim，避免用户不小心输空格
    const code: string = this.searchCode.trim()

    // 这里只支持 6 位代码（指数 / A 股都按 6 位走）
    if (code.length !== 6) {
      this.searchError = '请输入 6 位代码'
      return
    }

    // 进入请求态：展示“正在查询…”，并清掉上一次的错误提示
    this.searchLoading = true
    this.searchError = ''

    const httpRequest = http.createHttp()
    try {
      // 查询类型跟着当前模式走：指数模式就查指数接口，股票模式就查股票接口
      const type: 'index' | 'stock' = this.currentMode
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      // 200：后端返回单条行情
      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        const newItem: Quote = this.mapBackendQuote(backendItem, type)

        // 写入缓存：后面自选统计、列表行展示都依赖 quoteCache
        this.mergeQuotesToCache([newItem])

        // 搜索结果只显示“当前模式”的内容：
        // - 新结果放最前面
        // - 去掉重复（同 code + type 的只保留一条）
        const next: Quote[] = [newItem]
        for (let i: number = 0; i < this.searchedQuotes.length; i++) {
          const q: Quote = this.searchedQuotes[i]
          if (q.type !== type) continue
          if (q.code === newItem.code && q.type === newItem.type) continue
          next.push(q)
        }
        this.searchedQuotes = next
        return
      }

      // 404：后端明确表示没这个代码（给用户更友好的提示）
      if (res.responseCode === 404) {
        this.searchError = '未找到该代码'
      } else {
        // 其他错误：直接把 HTTP code 显示出来，便于排查
        this.searchError = `HTTP ${res.responseCode}`
      }
    } catch (e) {
      // 网络异常 / JSON 解析异常等：转成字符串显示
      this.searchError = `${e}`
    } finally {
      // 无论成功失败都要释放请求对象，并退出 loading 状态
      httpRequest.destroy()
      this.searchLoading = false
    }
  }

  // ===== 计算整段 K 线区域总宽度 =====
  private getKlineTotalWidth(): number {
    const count: number = this.klineGeoms.length;
    if (count <= 0) {
      return 0;
    }
    return count * this.candleBodyWidth + (count - 1) * this.candleGap;
  }


  // 加载分组
  private async loadGroupsFromPrefs(): Promise<void> {
    // 进来先打个 log，主要是排查“为啥分组没加载/加载慢/重复加载”这类问题
    this.dbg('loadGroupsFromPrefs:enter')

    // 打开本地 Preferences 文件（相当于一个小型的 key-value 存储）
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE)

    // 从指定 key 读出分组的 JSON 字符串；没有就给空串
    const jsonStr: string = (await pref.get(this.PREF_KEY_GROUPS, '')) as string

    // 第一次安装 / 没存过分组：走初始化逻辑，给一个“默认分组”
    if (jsonStr.trim().length === 0) {
      const g: StockGroup = {
        id: 1,
        name: '默认分组',
        stocks: [],
        createdAt: Date.now(),
        version: 0
      }

      // groups 用新数组直接覆盖，保证 UI 能立刻渲染出来
      this.groups = [g]
      this.activeGroupId = 1
      this.nextGroupId = 2 // 下次新建分组从 2 开始
      await this.saveGroupsToPrefs() // 把默认分组写回去，后续启动就能直接读到

      // 默认分组也要同步一次 activeStocks，否则自选页可能还是空状态
      this.syncActiveStocks()
      this.dbg('loadGroupsFromPrefs:initDefault')
      return
    }

    // 有数据：把 JSON 解析成 StockGroup[]（safeParseGroups 内部做过结构兜底）
    const parsed: StockGroup[] = this.safeParseGroups(jsonStr)
    this.groups = parsed

    // 扫一遍已有分组 id，用来计算 nextGroupId（避免新建分组 id 撞车）
    let maxId: number = 0
    for (let i: number = 0; i < parsed.length; i++) {
      if (parsed[i].id > maxId) maxId = parsed[i].id
    }
    this.nextGroupId = maxId + 1

    // 默认选中第一个分组（没有分组就设为 -1）
    this.activeGroupId = parsed.length > 0 ? parsed[0].id : -1

    // activeGroupId 变了就同步 activeStocks，保证自选列表/统计跟着刷新
    this.syncActiveStocks()
    this.dbg('loadGroupsFromPrefs:done')
  }


  // 把当前的分组列表（this.groups）序列化成 JSON，并写入 Preferences，保证下次启动还能恢复分组数据
  private async saveGroupsToPrefs(): Promise<void> {
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE);
    const jsonStr: string = JSON.stringify(this.groups);
    await pref.put(this.PREF_KEY_GROUPS, jsonStr);
    await pref.flush();
  }

  // 从 Preferences 里读出来的 JSON 字符串还原成 StockGroup[]，顺便做一次结构兜底（防止老数据/脏数据把页面搞崩）
  private safeParseGroups(jsonStr: string): StockGroup[] {
    try {
      // 先把字符串 parse 成 JS 对象；这里不做 any/unknown，统一按 Object 接住再逐字段转
      const rawObj: Object = JSON.parse(jsonStr) as Object;
      if (!Array.isArray(rawObj)) {
        // 不是数组就直接认为无效（我们存的 groups_json 本来就是数组）
        return [];
      }

      const arr: Object[] = rawObj as Object[];
      const result: StockGroup[] = [];

      for (let i: number = 0; i < arr.length; i++) {
        // 单个分组对象：用 Record<string, Object> 来做字段访问（符合 arkts6 的限制）
        const item: Record<string, Object> = arr[i] as Record<string, Object>;

        // 这些字段都走 Number/String 强转：就算缺字段/类型不对，也不会直接抛异常
        const id: number = Number(item['id']);
        const name: string = String(item['name'] ?? '');
        const createdAt: number = Number(item['createdAt'] ?? Date.now());
        const version: number = Number(item['version'] ?? 0); // 老版本没 version 的话，默认从 0 开始

        // stocks 可能不存在/不是数组：统一兜底成空数组
        const stocksRaw: Object = item['stocks'] ?? [];
        const stocks: StockRef[] = [];

        if (Array.isArray(stocksRaw)) {
          const sr: Object[] = stocksRaw as Object[];
          for (let j: number = 0; j < sr.length; j++) {
            const s: Record<string, Object> = sr[j] as Record<string, Object>;

            // 只接受 6 位 code；name 可选，有就带上
            const code: string = String(s['code'] ?? '').trim();
            if (code.length === 6) {
              const ref: StockRef = { code: code };

              const nm: string = String(s['name'] ?? '').trim();
              if (nm.length > 0) {
                ref.name = nm;
              }
              stocks.push(ref);
            }
          }
        }

        // 最起码要有 id + name 才算一个合法分组
        if (!Number.isNaN(id) && name.length > 0) {
          result.push({ id, name, stocks, createdAt, version });
        }
      }

      return result;
    } catch (e) {
      // JSON 解析失败或者结构特别离谱：直接回空，避免影响页面启动
      return [];
    }
  }

  // 找group id
  private findGroupIndex(groupId: number): number {
    for (let i: number = 0; i < this.groups.length; i++) {
      if (this.groups[i].id === groupId) {
        return i;
      }
    }
    return -1;
  }

  // 新建一个自选分组：写入内存（this.groups）+ 设为当前激活分组 + 落盘到 Preferences
  private async createGroup(name: string): Promise<void> {
    // 输入框里可能带空格，先 trim 一下；空名字就直接忽略
    const n: string = name.trim();
    if (n.length === 0) return;

    // 生成一个新的分组对象：id 用自增的 nextGroupId，version 从 0 开始（用于 List 强制刷新）
    const g: StockGroup = { id: this.nextGroupId, name: n, stocks: [], createdAt: Date.now(), version: 0 };
    this.nextGroupId += 1; // 下一个分组的 id 预留出来

    // 把新分组放到最前面（用户一般希望刚建的组马上出现在顶部）
    const copy: StockGroup[] = [g, ...this.groups];
    this.groups = copy;

    // 立即切到新分组，后续 UI 都以 activeGroupId 为准
    this.activeGroupId = g.id;

    // 关键：activeGroupId 变了以后，要把“当前活跃股票列表/统计”这条链路同步一遍
    // 不然 UI 会出现“看起来切了组，但数据还停留在上一个组”的情况
    this.syncActiveStocks();           // activeStocks / 股票数量等立刻更新
    this.refreshActiveGroupStats();    // 统计卡立刻刷新（新建空组就是 0）

    // 落盘：下次启动还能恢复分组
    await this.saveGroupsToPrefs();

    // 预热一下当前分组的行情缓存（新组为空也没关系，流程保持一致）
    await this.preloadActiveGroupQuotes();

    // 兜底再刷一次统计（有些情况下 preload 会改变缓存命中情况）
    this.refreshActiveGroupStats();
  }

  // 删除一个自选分组：从 this.groups 里移除 + 处理当前激活分组的切换 + 落盘保存
  private async deleteGroup(groupId: number): Promise<void> {
    // 1) 先把要删的分组过滤掉（不用 filter 是为了少踩 ArkTS 类型坑）
    const copy: StockGroup[] = [];
    for (let i: number = 0; i < this.groups.length; i++) {
      if (this.groups[i].id !== groupId) copy.push(this.groups[i]);
    }
    this.groups = copy;

    // 2) 如果删掉的是当前正在看的分组：需要把 activeGroupId 切到一个“还存在”的分组
    //    这里简单处理：切到列表第一个；如果一个都没了，就用 -1 表示“没有激活分组”
    if (this.activeGroupId === groupId) {
      this.activeGroupId = this.groups.length > 0 ? this.groups[0].id : -1;
    }

    // 3) 分组变动会影响 activeStocks / 统计卡，所以要立刻同步一下 UI 侧的“active 链路”
    //    否则会出现：分组已经删了，但页面还在显示旧分组的股票和统计
    this.syncActiveStocks();
    this.refreshActiveGroupStats();

    // 4) 落盘保存：保证下次启动分组列表是对的
    await this.saveGroupsToPrefs();

    // 5) 如果当前还有激活分组：预热一下该分组的行情缓存，然后再刷一次统计（走的是最新缓存/最新行情）
    if (this.activeGroupId >= 0) {
      await this.preloadActiveGroupQuotes();
      this.refreshActiveGroupStats();
    }
  }

  // 往指定分组里批量添加股票代码：做 6 位校验 + 去重 + 更新 groups + 落盘 +（必要时）预热行情
  private async addStocksToGroup(groupId: number, codes: string[]): Promise<void> {
    // 仅用于你自己的调试日志：方便在控制台快速确认传参和流程有没有走到
    this.dbg(`addStocksToGroup:enter gid=${groupId} codes=${JSON.stringify(codes)}`)

    // 找到目标分组在 this.groups 里的下标；没找到就直接退出
    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) {
      return
    }

    // 1) 先把当前分组已有的 code 收集起来：后面用来做去重
    const existed: Set<string> = new Set<string>()
    const curStocks: StockRef[] = this.groups[idx].stocks
    for (let i: number = 0; i < curStocks.length; i++) {
      existed.add(curStocks[i].code)
    }

    // 2) 处理本次要加的 codes：
    //    - trim 掉空格
    //    - 只接受 6 位
    //    - 跟分组已有的去重
    //    - 同一批 codes 自己也去重（比如用户一次输入重复了）
    const addList: StockRef[] = []
    for (let i: number = 0; i < codes.length; i++) {
      const code6: string = codes[i].trim()
      if (code6.length !== 6) {
        continue
      }
      if (existed.has(code6)) {
        continue
      }
      existed.add(code6)
      // 这里只先存 code；name 可以后面通过行情接口补全，减少一次输入负担
      addList.push({ code: code6 })
    }

    // 如果这次没有任何有效新增，就不用继续做后面的 UI 更新和落盘了
    if (addList.length === 0) {
      this.dbg('addStocksToGroup:skip empty addList')
      return
    }

    // 3) 更新 groups：
    //    这里必须“新数组 + 新对象”，ArkUI 才更容易识别到结构变化并触发刷新
    const newGroups: StockGroup[] = this.groups.slice()
    const g: StockGroup = newGroups[idx]
    const newStocks: StockRef[] = g.stocks.concat(addList)
    // cloneGroupWithStocks 一般会顺带 version+1，避免 List 因为引用没变而不刷新
    newGroups[idx] = this.cloneGroupWithStocks(g, newStocks)
    this.groups = newGroups

    // 4) 如果改的是当前正在看的分组：马上同步 activeStocks
    //    不然用户会感觉“我点了添加，但列表没变/统计没变”
    if (this.activeGroupId === groupId) {
      this.syncActiveStocks()
    }

    // 5) 给个简单提示：告诉用户这次加了多少只、当前分组总共有多少只
    this.watchlistMsg = `已添加 ${addList.length} 只，当前 ${newGroups[idx].stocks.length} 只`

    // 6) 落盘保存：保证下次启动分组数据还在
    await this.saveGroupsToPrefs()
    this.dbg('addStocksToGroup:afterSavePrefs')

    // 7) 如果是当前分组：顺手预热一下这些股票的行情
    //    这样 quoteCache 很快就有数据，统计卡（涨跌/均值）就不会一直是 "--"
    if (this.activeGroupId === groupId) {
      await this.preloadActiveGroupQuotes()
      this.refreshActiveGroupStats()
      this.dbg('addStocksToGroup:afterPreload')
    }
  }

  // 从指定分组里批量移除股票：把要删的 code 做成集合，然后重建 stocks、更新 groups、落盘，并同步当前分组的 UI/统计
  private async removeStocksFromGroup(groupId: number, codes: string[]): Promise<void> {
    // 调试用：确认是谁触发的、传进来要删哪些代码
    this.dbg(`removeStocksFromGroup:enter gid=${groupId} codes=${JSON.stringify(codes)}`)

    // 找目标分组在 groups 里的位置；找不到就不继续了
    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) {
      return
    }

    // 1) 先把“要移除的 code”整理成 Set：
    //    - trim 空格
    //    - 只认 6 位
    //    - Set 天生去重，避免重复删同一个 code 还要额外判断
    const toRemove: Set<string> = new Set<string>()
    for (let i: number = 0; i < codes.length; i++) {
      const code6: string = codes[i].trim()
      if (code6.length === 6) {
        toRemove.add(code6)
      }
    }

    // 如果没有任何有效的 code，就直接退出，避免做无意义的状态更新
    if (toRemove.size === 0) {
      this.dbg('removeStocksFromGroup:skip empty toRemove')
      return
    }

    // 2) 用过滤的方式生成新的 stocks（把要删的过滤掉）
    const oldStocks: StockRef[] = this.groups[idx].stocks
    const newStocks: StockRef[] = []
    for (let i: number = 0; i < oldStocks.length; i++) {
      const code: string = oldStocks[i].code
      if (!toRemove.has(code)) {
        // 这里直接复用原来的 StockRef（里面可能带着 name），不做额外改动
        newStocks.push(oldStocks[i])
      }
    }

    // 如果过滤完长度没变，说明这次要删的 codes 都不在分组里，就别折腾 UI/落盘了
    if (newStocks.length === oldStocks.length) {
      this.dbg('removeStocksFromGroup:skip no changes')
      return
    }

    // 3) 更新 groups：同样走“新数组 + 新对象”的方式，保证 UI 能刷新出来
    const newGroups: StockGroup[] = this.groups.slice()
    const g: StockGroup = newGroups[idx]
    newGroups[idx] = this.cloneGroupWithStocks(g, newStocks)
    this.groups = newGroups

    // 4) 如果删的是当前正在看的分组：马上同步 activeStocks
    //    不然用户会看到分组里少了，但下面列表/统计还停留在旧数据
    if (this.activeGroupId === groupId) {
      this.syncActiveStocks()
    }

    // 5) 简单提示：告诉用户这次删了多少、现在还剩多少
    this.watchlistMsg = `已删除 ${oldStocks.length - newStocks.length} 只，当前 ${newGroups[idx].stocks.length} 只`

    // 6) 落盘保存：下次启动依旧保持删除后的状态
    await this.saveGroupsToPrefs()
    this.dbg('removeStocksFromGroup:afterSavePrefs')

    // 7) 如果是当前分组：可选地再预热一次行情并刷新统计
    //    （删除后 cache 里可能还有旧标的，但统计是按 activeStocks 算的，所以这里主要是保持流程一致）
    if (this.activeGroupId === groupId) {
      await this.preloadActiveGroupQuotes()
      this.refreshActiveGroupStats()
      this.dbg('removeStocksFromGroup:afterPreload')
    }
  }

  // 用“新对象”克隆一个分组，并替换 stocks。
  // 这里顺手把 version +1：主要是给 List/渲染层一个“我真的变了”的信号，避免某些情况下 UI 认为结构没变而不刷新。
  private cloneGroupWithStocks(g: StockGroup, stocks: StockRef[]): StockGroup {
    return {
      id: g.id,
      name: g.name,
      stocks: stocks,          // 新的股票列表（已经 add/remove 处理过）
      createdAt: g.createdAt,  // 分组创建时间不变
      version: g.version + 1   // 每次改 stocks 都 +1，用来强制触发 UI 重新渲染
    };
  }

  // 取“当前选中的分组”对象。
  // 这里返回 StockGroup | null：外面用的时候可以很直观地做空判断，避免 activeGroupId 还没初始化/分组被删掉时直接崩。
  private getActiveGroup(): StockGroup | null {
    // activeGroupId 为 -1 这类值时，表示当前没有选中任何分组
    if (this.activeGroupId < 0) {
      return null;
    }

    // 用 id 找到它在 this.groups 数组里的下标
    const idx: number = this.findGroupIndex(this.activeGroupId);
    if (idx < 0) {
      // 没找到：一般是分组刚被删除、或本地数据异常导致
      return null;
    }

    // 找到了就直接返回对应分组
    return this.groups[idx];
  }


  private findQuoteInCaches(code: string): Quote | null {
    // 传进来的可能带空格（比如输入框/剪贴板），先做一次标准化
    const code6: string = code.trim();
    // 我们的后端/数据约定都是 6 位代码；不符合就直接当作无效输入
    if (code6.length !== 6) return null;

    // 缓存里的 key 统一做了前缀区分：stock-xxxxxx / index-xxxxxx
    // 这里先按股票查一次（大部分场景用户更常选股票）
    const kStock: string = `stock-${code6}`;
    const vStock: Quote | undefined = this.quoteCache.get(kStock);
    if (vStock !== undefined) return vStock;

    // 股票没找到，再按指数查一次
    const kIndex: string = `index-${code6}`;
    const vIndex: Quote | undefined = this.quoteCache.get(kIndex);
    if (vIndex !== undefined) return vIndex;

    // 两类都没有，说明缓存里确实没有这个标的
    return null;
  }

  private getStockRowName(s: StockRef): string {
    // 这里“读一下 quoteCacheVer”是为了让 UI 依赖到缓存版本号：
    // 只要行情缓存被更新并且你递增了 quoteCacheVer，这一行就会重新计算、重新渲染名称。
    const _qv: number = this.quoteCacheVer

    // 优先从缓存里拿“最新行情对应的名字”（最权威，能自动跟随后端修正/补全）
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    if (q !== null) {
      return q.name;
    }

    // 缓存没命中时，退一步用 StockRef 里保存过的 name（比如第一次添加自选时写进去的）
    if (s.name && s.name.length > 0) {
      return s.name;
    }

    // 两边都没有就给个占位符，避免 UI 显示空字符串
    return '—';
  }

  // 用于在自选列表里展示某只股票的“当前价”文本（优先用缓存行情，没有就给占位符）
  private getStockRowPriceText(s: StockRef): string {
    // 同 getStockRowName：读一下 quoteCacheVer，确保缓存版本变化时这一行会触发重算/刷新
    const _qv: number = this.quoteCacheVer

    // 尝试从行情缓存里拿到该股票的最新报价
    const q: Quote | null = this.findStockQuoteInCache(s.code)

    // 有行情：按 2 位小数展示；没行情：用占位符，避免出现 NaN/空白
    return (q !== null) ? q.price.toFixed(2) : '--'
  }

  // 用于在自选列表里展示某只股票的“涨跌额”文本（优先用缓存行情，没有就给占位符）
  private getStockRowChangeText(s: StockRef): string {
    // 读一下 quoteCacheVer，确保缓存行情更新时这一行会跟着刷新
    const _qv: number = this.quoteCacheVer

    // 尝试从行情缓存里拿到该股票的最新报价
    const q: Quote | null = this.findStockQuoteInCache(s.code)

    // 有行情：按 2 位小数展示涨跌额；没行情：用占位符
    return (q !== null) ? q.change.toFixed(2) : '--'
  }

  // 用于在自选列表里展示某只股票的“涨跌幅(%)”文本（优先用缓存行情，没有就给占位符）
  private getStockRowPctText(s: StockRef): string {
    // 读一下 quoteCacheVer，确保缓存行情更新时这一行会跟着刷新
    const _qv: number = this.quoteCacheVer

    // 尝试从行情缓存里拿到该股票的最新报价
    const q: Quote | null = this.findStockQuoteInCache(s.code)

    // 有行情：保留 2 位小数并补上 %；没行情：用占位符
    return (q !== null) ? (q.changePercent.toFixed(2) + '%') : '--'
  }

  // 用于在自选列表里决定“价格数字”的颜色：涨(含0)用红色，跌用绿色；没拿到行情就用灰色占位
  private getStockRowPriceColor(s: StockRef): string {
    // 读一下 quoteCacheVer，确保行情缓存更新时，这个颜色也会触发重新计算/渲染
    const _qv: number = this.quoteCacheVer

    // 从缓存里取这只股票的最新行情（拿不到就返回 null）
    const q: Quote | null = this.findStockQuoteInCache(s.code)

    // 没行情：统一用灰色，避免 UI 出现“红绿乱跳”
    if (q === null) {
      return '#AAAAAA'
    }

    // 有行情：按涨跌决定颜色（change >= 0 也算“涨/平”，走红色）
    return (q.change >= 0) ? '#FF4D4F' : '#39D98A'
  }

  // 用于在自选列表里决定“副文本/辅助信息”（比如涨跌额、涨跌幅一类）显示的颜色：涨(含0)用浅红，跌用浅绿；没拿到行情就用更暗的灰
  private getStockRowSubColor(s: StockRef): string {
    // 读一下 quoteCacheVer，确保行情缓存更新时，这个颜色也会跟着触发重新计算/渲染
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code)

    // 没有行情数据：副文本用深一点的灰色，和主价格的灰区分开
    if (q === null) {
      return '#777777'
    }

    // 有行情：涨用浅红、跌用浅绿（比主价格色更“柔和”，不抢主视觉）
    return (q.change >= 0) ? '#FF7875' : '#6BE49B'
  }

  // 确保某个股票 code 的行情已经进了 quoteCache：如果缓存里没有，就补一次 by_code 请求把它拉进来（主要给自选统计/列表兜底用）
  private async ensureQuoteCachedForCode(code: string): Promise<void> {
    const code6: string = code.trim()
    if (code6.length !== 6) return

    // 这里只看 stock-${code} 这一条缓存：避免“指数/股票同码”时，被 index 缓存误判成已经有数据
    const cacheKey: string = `stock-${code6}`
    const existed: Quote | undefined = this.quoteCache.get(cacheKey)
    if (existed !== undefined) return

    const httpRequest = http.createHttp()
    try {
      // 补拉单个股票行情：只要成功写入 cache，后续统计/列表就能用缓存值算出来
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=stock&code=${code6}`
      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      this.dbg(`ensureQuoteCachedForCode:request ${code6}`)
      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        const q: Quote = this.mapBackendQuote(backendItem, 'stock')

        // mergeQuotesToCache 内部会更新 quoteCache + quoteCacheVer，
        // quoteCacheVer 变化会让依赖缓存的 UI（比如统计卡）刷新
        this.mergeQuotesToCache([q])
        this.dbg(`ensureQuoteCachedForCode:cached ${code6}`)
      }
    } catch (e) {
      // 这里是“兜底补拉”，失败就算了，不要影响主流程（UI 仍然可以显示 '--'）
    } finally {
      httpRequest.destroy()
    }
  }

  // 按 type（index/stock）兜底补拉单个标的行情：如果 quoteCache 里还没有，就调用 by_code 接口拉一次并写入缓存
  private async ensureQuoteCachedByType(type: QuoteType, code6: string): Promise<void> {
    const code: string = code6.trim()
    if (code.length !== 6) return

    // 缓存 key 统一用 `${type}-${code}`：同一代码在指数/股票下也能共存，不会互相覆盖
    const cacheKey: string = `${type}-${code}`
    const existed: Quote | undefined = this.quoteCache.get(cacheKey)
    if (existed !== undefined) return

    const httpRequest = http.createHttp()
    try {
      // 只补拉这一条：用于“列表/统计/卡片”在缺行情时的兜底，不影响主 fetchQuotes 流程
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`
      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendQuote = JSON.parse(res.result as string) as BackendQuote

        // 后端字段映射 + 填上 type，保持 Quote 结构一致（指数/股票一套逻辑）
        const q: Quote = this.mapBackendQuote(backendItem, type)

        // 写入 quoteCache；mergeQuotesToCache 通常会顺带 bump quoteCacheVer，触发依赖缓存的 UI 刷新
        this.mergeQuotesToCache([q])
      }
    } catch (e) {
      // 兜底补拉失败就忽略：最多让 UI 显示 '--'，不要把错误扩散到页面主流程
    } finally {
      httpRequest.destroy()
    }
  }

  // 按 type（index/stock）+ code 从 quoteCache 里取一条行情；取不到就返回 null
  private findQuoteInCacheByType(type: QuoteType, code6: string): Quote | null {
    const code: string = code6.trim()
    if (code.length !== 6) return null

    // 缓存 key 规则：`${type}-${code}`，确保指数和股票同码也不会互相顶掉
    const k: string = `${type}-${code}`

    // Map.get 取不到会是 undefined，这里统一转成 null，外面写判断更顺手
    const v: Quote | undefined = this.quoteCache.get(k)
    return (v !== undefined) ? v : null
  }

  // 预热当前分组里所有股票的行情到缓存（quoteCache），让自选列表/统计卡能尽快拿到最新数据
  private async preloadActiveGroupQuotes(): Promise<void> {
    const g: StockGroup | null = this.getActiveGroup()
    if (g === null) {
      // 没有选中分组（或分组不存在）就不用做任何事
      return
    }

    // 每只股票对应一个“确保缓存存在”的请求任务
    const tasks: Promise<void>[] = []
    for (let i: number = 0; i < g.stocks.length; i++) {
      const code: string = g.stocks[i].code
      tasks.push(this.ensureQuoteCachedForCode(code))
    }

    // 并行预热：某一只失败也不影响其它（避免因为单个接口异常导致整组卡住）
    await Promise.all(tasks.map((p: Promise<void>) => p.catch(() => { })))
  }

  // 把最新拉到的行情合并进本地 quoteCache（key = `${type}-${code}`），同时触发统计刷新
  private mergeQuotesToCache(list: Quote[]): void {
    console.info(`[DBG] mergeQuotesToCache: in list=${list.length} oldCache=${this.quoteCache.size}`);

    // 用“新 Map”拷贝一份旧缓存：避免直接在原 Map 上改导致状态变更不明显（也方便统一替换）
    const newMap: Map<string, Quote> = new Map<string, Quote>();
    this.quoteCache.forEach((v: Quote, k: string) => {
      newMap.set(k, v);
    });

    // 把这次请求回来的行情逐条写入缓存：同 key 会覆盖旧值（相当于更新）
    for (let i: number = 0; i < list.length; i++) {
      const q: Quote = list[i];
      const key: string = `${q.type}-${q.code}`; // stock-300750 / index-000001 这种，避免同码冲突
      newMap.set(key, q);
    }

    // 一次性替换缓存引用，配合版本号让依赖方（统计/列表）知道“缓存更新过了”
    this.quoteCache = newMap;
    this.quoteCacheVer += 1;

    // 行情一变，自选分组的涨跌统计（up/down/avg）也要跟着立刻刷新
    this.refreshActiveGroupStats();

    console.info(`[DBG] mergeQuotesToCache: out newCache=${this.quoteCache.size}`);
  }

  // 取当前“选中的分组”（activeGroupId 对应的那一个）；如果当前没有选中分组或找不到，就返回 null
  private getActiveGroupOrNull(): StockGroup | null {
    // activeGroupId < 0 表示当前没选中任何分组（比如刚启动还没初始化完，或已经删光了）
    if (this.activeGroupId < 0) return null

    // 在 groups 数组里找 activeGroupId 对应的下标
    const idx: number = this.findGroupIndex(this.activeGroupId)

    // 没找到说明数据不同步/分组被删了，兜底返回 null
    if (idx < 0) return null

    // 找到就直接返回这个分组对象
    return this.groups[idx]
  }

  // 顶部/统计区用的：拿到当前选中分组的名称，没选中就返回空字符串
  private getActiveGroupNameText(): string {
    // 先尝试取当前 active 分组对象（可能为 null）
    const g: StockGroup | null = this.getActiveGroupOrNull()

    // 有分组就显示它的 name；没有就显示空（UI 一般会自然不展示）
    return g ? g.name : ''
  }

  // 返回当前选中分组的股票列表（用于渲染“自选列表”等 UI）；如果当前没有有效分组就返回空数组
  private getActiveGroupStocksList(): StockRef[] {
    // 先拿到当前 activeGroupId 对应的分组对象（可能为空）
    const g: StockGroup | null = this.getActiveGroupOrNull()

    // 有分组就返回它的 stocks；没有就返回空数组，避免 UI/调用方再做空判断
    return g ? g.stocks : []
  }

  // 根据“当前分组里的股票列表 + 已缓存的行情数据”，计算分组统计卡要显示的几个数字（总数/涨/跌/平均涨跌幅）
  private computeStatsFromStocks(list: StockRef[]): ActiveGroupStats {
    // 分组里一共多少只股票：这个不依赖行情缓存，直接用列表长度就行
    const stockCount: number = list.length

    // up/down：统计今天涨/跌的数量（只统计拿得到行情的那部分）
    let up: number = 0
    let down: number = 0

    // 平均涨跌幅：累加 percent，最后除以有效数量
    let sumPct: number = 0
    let cntPct: number = 0

    for (let i: number = 0; i < list.length; i++) {
      const code: string = list[i].code

      // 从缓存里找这只股票的行情（这里只算股票，所以用 findStockQuoteInCache，避免 index 同码干扰）
      // 如果这只股票还没拉到行情（缓存里没有），就先跳过，不把它算进平均值
      const q: Quote | null = this.findStockQuoteInCache(code)

      if (q !== null) {
        // change >= 0 当作“上涨/不跌”（包含 0 的情况）
        if (q.change >= 0) {
          up += 1
        } else {
          down += 1
        }

        // 只要有行情，就把涨跌幅纳入平均值的统计
        sumPct += q.changePercent
        cntPct += 1
      }
    }

    // 如果一只行情都没拿到，就显示 “--”，避免出现 NaN%
    const avgPct: string = (cntPct > 0) ? ((sumPct / cntPct).toFixed(2) + '%') : '--'

    // 统计卡那边用的是 string，所以这里统一转成字符串返回
    return {
      stockCount: stockCount.toString(),
      up: up.toString(),
      down: down.toString(),
      avgPct: avgPct
    }
  }

  // 只从缓存里找“股票”行情：给自选分组用，避免 index/stock 代码相同（比如 000001）时拿错数据
  private findStockQuoteInCache(code: string): Quote | null {
    // 先做一次最基本的清洗，避免用户输入/存储里带空格
    const code6: string = code.trim()
    if (code6.length !== 6) return null  // 不是 6 位代码就直接当无效

    // 缓存的 key 统一用 `${type}-${code}`，这里固定查 stock
    const kStock: string = `stock-${code6}`

    // Map.get 取不到会返回 undefined，这里显式处理成 null，外层判断更直观
    const v: Quote | undefined = this.quoteCache.get(kStock)
    return (v !== undefined) ? v : null
  }

  private logSeq: number = 0;

  // 统一的调试日志：把当前页面关键状态打一行出来，方便你追“分组 / 缓存 / UI 切换”到底哪里没同步
  private dbg(tag: string): void {
    // 给每条日志一个递增序号，排查异步流程时更容易按顺序对齐
    this.logSeq += 1;

    // 当前选中的分组 id（-1 表示还没选中/没有分组）
    const gid: number = this.activeGroupId;

    // activeGroupId 在 groups 里的下标；找不到就会是 -1
    const idx: number = this.findGroupIndex(gid);

    // 默认先给“无效值”，避免 idx=-1 时访问数组越界
    let activeCount: number = -1;
    let activeName: string = '';

    // 只有 idx 合法时才读取分组信息（分组名、分组里股票数量）
    if (idx >= 0) {
      activeCount = this.groups[idx].stocks.length;
      activeName = this.groups[idx].name;
    }

    // 一条日志尽量把你排查时最常看的状态都带上：
    // - 当前在哪个 Tab / 行情模式（指数 or 股票）
    // - 当前 activeGroupId 是否有效、对应的分组名、分组里有多少只
    // - groups 总数、缓存条目数、activeStocks 数量、缓存版本号（用来确认 UI 是否应该刷新）
    console.info(
      `[DBG#${this.logSeq}] ${tag} | tab=${this.currentTabIndex} mode=${this.currentMode}` +
        ` | activeGroupId=${gid} activeIdx=${idx} activeName=${activeName} activeStocks=${activeCount}` +
        ` | groups=${this.groups.length} cache=${this.quoteCache.size}` +
        ` | activeStocks=${this.activeStocks.length} cache=${this.quoteCache.size} cacheVer=${this.quoteCacheVer}`
    );
  }

  // 同步“当前选中分组”的股票列表到 activeStocks（UI 列表、统计卡都走这条链路）
  private syncActiveStocks(): void {
    // 先拿到当前 activeGroupId 对应的分组；拿不到就当作没有选中分组
    const g: StockGroup | null = this.getActiveGroupOrNull()

    if (g === null) {
      // 没有有效分组：自选列表清空
      this.activeStocks = []
    } else {
      // 这里用 slice() 是为了拿到一个“新的数组引用”
      // ArkUI 的状态更新更依赖引用变化：直接用 g.stocks 可能导致 UI 认为“还是同一个数组”而不刷新
      this.activeStocks = g.stocks.slice()
    }

    // 用一个版本号配合 key/依赖，强制相关 List 或统计区域在需要时重新渲染
    this.activeStocksVer += 1

    // activeStocks 变了，顺手把统计（数量 / 涨跌数 / 平均涨幅）也刷新一下
    this.refreshActiveGroupStats()
  }

  // 返回某个分组的“平均涨跌幅”展示文本（用于分组列表里的那一行数据）
  private getGroupAvgPctText(groupId: number): string {
    // 这里故意读一下 quoteCacheVer，让 ArkUI 知道这个返回值依赖行情缓存。
    // 行情一更新（mergeQuotesToCache 里 quoteCacheVer++），分组行里的平均涨幅就会跟着重算/重绘。
    const _qv: number = this.quoteCacheVer

    // 先找到分组在 groups 数组里的位置；找不到就给一个占位文本
    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) return '--'

    // 统计是“现算现用”：用该分组的 stocks 列表去缓存里找最新 quote，再算平均涨跌幅
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.groups[idx].stocks)
    return st.avgPct
  }

  // 计算某个分组“平均涨跌幅”对应的展示颜色：涨用红色、跌用绿色、没数据用灰色
  private getGroupAvgPctColor(groupId: number): string {
    // 读一下缓存版本号，让 ArkUI 知道：行情缓存变了，这个颜色也要跟着重新算/刷新
    const _qv: number = this.quoteCacheVer

    // 根据 groupId 找到分组下标；找不到就给一个默认灰色
    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) return '#777777'

    // 复用现成的统计逻辑，拿到该分组的 avgPct（形如 "1.23%" 或 "--"）
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.groups[idx].stocks)
    const avg: string = st.avgPct

    // 没有可用行情时 avgPct 会是 "--"，这时候用灰色
    if (avg === '--') {
      return '#777777'
    }

    // avgPct 是字符串带 "%"，这里去掉百分号后转成数字，方便判断正负
    const v: number = Number(avg.replace('%', ''))
    if (Number.isNaN(v)) {
      return '#777777'
    }

    // v >= 0 认为是上涨（红），v < 0 认为是下跌（绿）
    return v >= 0 ? '#FF4D4F' : '#39D98A'
  }


  // 把一条行情 Quote 整理成“桌面卡片”需要的绑定数据（主要是把数值转成可直接展示的字符串）
  private buildWidgetDataFromQuote(q: Quote): WidgetBindData {
    // 桌面卡片上通常会显示“更新时间”，这里用本地时间拼一个 HH:mm
    const now: Date = new Date()
    const hh: string = now.getHours().toString().padStart(2, '0')
    const mm: string = now.getMinutes().toString().padStart(2, '0')

    return {
      // 当前这张卡片的实例 id（从 AppStorageV2 消费过来），后面更新卡片时要靠它定位到哪张卡
      formId: this.activeFormId,

      // 卡片标题：指数 / 股票（给用户一眼分清楚当前绑定的是哪种标的）
      title: (q.type === 'index') ? '指数' : '股票',

      // 标的基本信息：代码、名称
      code: q.code,
      name: q.name,

      // 价格/涨跌/涨跌幅都转成“直接可显示”的文本，避免卡片侧再做格式化
      priceText: q.price.toFixed(2),
      changeText: q.change.toFixed(2),
      pctText: q.changePercent.toFixed(2) + '%',

      // 更新时间（只做到分钟级别，足够让人判断是不是刚刷新）
      tsText: `${hh}:${mm}`,

      // 这两个开关一般用于卡片上的“上一只/下一只”按钮是否可点（你这里先固定为 true）
      hasPrev: true,
      hasNext: true
    }
  }

  // 如果这次是从桌面卡片点进来的（activeFormId 有值），就把“当前选中的标的”回写到那张卡片上
  private async updateWidgetIfLaunchedFromForm(q: Quote): Promise<void> {
    // formId 可能带空格，先 trim 一下，避免更新失败还不好排查
    const fid: string = this.activeFormId.trim()

    hilog.error(DOMAIN, TAG, '[WIDGET] ENTER updateWidget fid=%{public}s code=%{public}s', fid, q.code)

    // 没有 formId 说明不是从卡片启动的，或者 formId 还没消费到，直接跳过
    if (fid.length === 0) {
      hilog.error(DOMAIN, TAG, '[WIDGET] SKIP activeFormId empty')
      return
    }

    try {
      // 把当前 Quote 转成卡片侧能直接展示的绑定数据（字符串化、带时间戳等）
      const data: WidgetBindData = this.buildWidgetDataFromQuote(q)

      // FormKit 需要的是 FormBindingData，这里把普通对象包装一下
      const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(data)

      // 更新桌面上那张 formId 对应的卡片内容
      await formProvider.updateForm(fid, binding)

      hilog.error(DOMAIN, TAG, '[WIDGET] OK updateForm formId=%{public}s code=%{public}s', fid, q.code)
    } catch (e) {
      // 更新失败通常是 formId 不对、卡片已被移除、或数据结构不匹配，这里打出来方便定位
      hilog.error(
        DOMAIN,
        TAG,
        '[WIDGET] FAIL updateForm formId=%{public}s err=%{public}s',
        fid,
        JSON.stringify(e)
      )
    }
  }

  // 从 AppStorageV2 里“拿一次”桌面卡片传来的 formId：同步到页面字段后立刻清空，避免重复触发
  private consumeFormIdFromAppStorageV2(from: string): void {
    const st: ActiveFormState | undefined =
      AppStorageV2.connect(ActiveFormState, APP_STORAGE_KEY_ACTIVE_FORM, () => new ActiveFormState())

    // 连接失败就直接返回（后续也没法同步桌面卡片）
    if (!st) {
      hilog.error(DOMAIN, TAG, '[%{public}s] AppStorageV2.connect failed', from)
      return
    }

    const fid: string = st.formId
    if (fid.length === 0) {
      // 不是从卡片进来的 / 已经被消费过
      hilog.info(DOMAIN, TAG, '[%{public}s] no formId in ActiveFormState', from)
      return
    }

    // 记到页面上，后面 updateForm 要用
    this.activeFormId = fid

    // 用完就清掉，避免 onPageShow / onResume 多次进来重复消费
    st.formId = ''

    hilog.info(DOMAIN, TAG, '[%{public}s] consumed formId=%{public}s', from, fid)
  }

  // 确保系统通知权限/开关已打开：没开就引导用户去系统界面打开，最后返回是否可用
  private async ensureNotificationEnabled(): Promise<boolean> {
    try {
      // 先查一次当前应用的通知开关状态（系统层面的总开关/权限）
      const enabled: boolean = await notificationManager.isNotificationEnabled()
      if (enabled) {
        return true
      }

      // 没开的话，拉起系统的通知开关/授权页面，让用户手动打开
      await notificationManager.requestEnableNotification()

      // 用户从系统页面返回后，再查一次，确认最终结果
      const enabled2: boolean = await notificationManager.isNotificationEnabled()
      return enabled2
    } catch (e) {
      // 这里不强行抛错：通知不可用就按不可用处理，避免影响主流程
      return false
    }
  }

  // 从本地偏好（Preferences）里把“通知规则列表”读出来，恢复到 notifyRules；顺便把 nextRuleId 校准到不会撞号
  private async loadNotifyRulesFromPrefs(): Promise<void> {
    // 打开（或创建）这个应用自己的偏好存储文件：PREF_FILE 是文件名，不是路径
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE)

    // 用 PREF_KEY_RULES 这把“键”取出之前保存的规则 JSON 字符串；取不到就给空串
    const jsonStr: string = (await pref.get(this.PREF_KEY_RULES, '')) as string

    // 为空说明以前没保存过规则：保持当前 @State notifyRules 默认值即可
    // 同时把 nextRuleId 按当前规则最大 id 重新算一遍，避免后续新增规则 id 冲突
    if (jsonStr.trim().length === 0) {
      this.recomputeNextRuleId()
      return
    }

    // 有内容则按我们的结构做一次“安全解析”（字段缺失/类型不对会被过滤）
    const parsed: NotifyRule[] = this.safeParseNotifyRules(jsonStr)

    // 解析出来确实有有效规则，才覆盖到页面状态；然后同样校准 nextRuleId
    if (parsed.length > 0) {
      this.notifyRules = parsed
      this.recomputeNextRuleId()
    }
  }


  // 把当前内存里的 notifyRules 序列化成 JSON，写回到 Preferences（下次启动/回到页面还能恢复）
  private async saveNotifyRulesToPrefs(): Promise<void> {
    // 打开（或创建）本应用的偏好存储文件（PREF_FILE 是文件名标识）
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE)

    // notifyRules 是对象数组，Preferences 只能存基础类型，所以这里转成 JSON 字符串再存
    const jsonStr: string = JSON.stringify(this.notifyRules)

    // 用 PREF_KEY_RULES 作为“键”把这串 JSON 保存进去（同一个键会覆盖旧值）
    await pref.put(this.PREF_KEY_RULES, jsonStr)

    // flush 才是把本次 put 的改动真正落盘（不 flush 可能只在内存里，重启后丢）
    await pref.flush()
  }


  // 从 Preferences 里读出来的是一串 JSON，这里负责“尽量安全地”把它还原成 NotifyRule[]（老数据缺字段也能兜住）
  private safeParseNotifyRules(jsonStr: string): NotifyRule[] {
    try {
      // 先把 JSON 解析成一个泛对象；后面会逐项做校验/兜底，所以这里不直接信它的结构
      const raw: Object = JSON.parse(jsonStr) as Object
      if (!Array.isArray(raw)) return []

      // raw 确认是数组后，再按数组去处理
      const arr: Object[] = raw as Object[]
      const out: NotifyRule[] = []

      for (let i: number = 0; i < arr.length; i++) {
        // 每个元素理论上都是一个对象：{ id, quoteType, stockCode, ... }
        // 用 Record<string, Object> 只是为了“能用 key 取值”，并不代表它一定合法
        const item: Record<string, Object> = arr[i] as Record<string, Object>

        // id：用 Number 强转；如果是 '123' 这种字符串也能转成 123
        const id: number = Number(item['id'])

        // stockCode / stockName / conditionText：统一转成字符串并 trim，避免 null/undefined 造成崩溃
        const stockCode: string = String(item['stockCode'] ?? '').trim()
        const stockName: string = String(item['stockName'] ?? '').trim()
        const conditionText: string = String(item['conditionText'] ?? '').trim()

        // enabled：Boolean(...) 会把 0/1、true/false、空字符串等转成布尔值
        const enabled: boolean = Boolean(item['enabled'])

        // quoteType：新增字段。老版本存的数据没有 quoteType，这里默认按 stock 处理
        // 同时做一次白名单：只有 'index' 才当成 index，其它一律按 stock（防脏数据）
        const qtRaw: string = String(item['quoteType'] ?? 'stock').trim()
        const quoteType: QuoteType = (qtRaw === 'index') ? 'index' : 'stock'

        // 最基本的合法性校验：id 必须能转成数字、代码必须 6 位、名称/条件不能为空
        // 注意：这里不校验 quoteType（上面已经做过白名单兜底了）
        if (!Number.isNaN(id) && stockCode.length === 6 && stockName.length > 0 && conditionText.length > 0) {
          // 通过校验的才加入输出数组，避免把脏数据带到 UI / RuleEngine
          out.push({
            id: id,
            quoteType: quoteType,
            stockCode: stockCode,
            stockName: stockName,
            conditionText: conditionText,
            enabled: enabled
          })
        }
      }
      return out
    } catch (e) {
      // JSON 解析失败/结构异常：直接返回空数组，保证页面不会因为坏数据崩
      return []
    }
  }


  // 根据当前已有的通知规则，重新计算 nextRuleId（保证新增规则时 id 不会和已有的重复）
  private recomputeNextRuleId(): void {
    // 从 0 开始扫一遍，找出当前 notifyRules 里最大的 id
    let maxId: number = 0
    for (let i: number = 0; i < this.notifyRules.length; i++) {
      // 只关心 id 最大值；这里不做复杂校验，因为 notifyRules 本身已经是我们维护的结构
      if (this.notifyRules[i].id > maxId) maxId = this.notifyRules[i].id
    }

    // 下一个可用 id = 当前最大 id + 1（简单、直观，也方便排查问题）
    this.nextRuleId = maxId + 1
  }

  // 从“通知”页的输入框组装一条新规则：校验输入 → 去重 → 补全名称 → 写入 notifyRules 并落盘
  private async addNotifyRuleFromUI(): Promise<void> {
    // 1) 读取并校验代码：规则引擎/后端接口都按 6 位代码走，长度不对直接拦住
    const code: string = this.ruleStockCodeInput.trim()
    if (code.length !== 6) {
      this.ruleMsg = '请输入 6 位代码'
      return
    }

    // 2) 读取并校验阈值：这里允许用户输入字符串，但最终必须能转成数字
    const v: number = Number(this.rulePriceInput.trim())
    if (Number.isNaN(v)) {
      this.ruleMsg = '请输入正确的价格阈值'
      return
    }

    // 3) 拼出统一的条件文案：后续“去重”和“命中判断”都依赖 conditionText 的一致性
    //    ruleOpIndex=0 表示 ≥，=1 表示 ≤（你界面上两个按钮就是在改这个）
    const opText: string = (this.ruleOpIndex === 0) ? '≥' : '≤'
    const cond: string = `价格 ${opText} ${v.toFixed(2)}`

    // 4) 去重：同 quoteType + 同 code + 同 cond，视为同一条规则，直接提示不再添加
    for (let i: number = 0; i < this.notifyRules.length; i++) {
      const r: NotifyRule = this.notifyRules[i]
      if (r.quoteType === this.ruleQuoteType && r.stockCode === code && r.conditionText === cond) {
        this.ruleMsg = '该规则已存在'
        return
      }
    }

    // 5) 尝试把标的名称补全出来：优先通过缓存/后端拿到 name，这样列表里显示更友好
    //    注意这里按 quoteType 请求，避免“指数/股票同码”时拿错名字
    const qt: QuoteType = this.ruleQuoteType
    await this.ensureQuoteCachedByType(qt, code)
    const q: Quote | null = this.findQuoteInCacheByType(qt, code)
    const name: string = (q !== null) ? q.name : code  // 拿不到就先用 code 顶上

    // 6) 组装新规则对象：enabled 默认 true，表示添加后立刻生效
    const newRule: NotifyRule = {
      id: this.nextRuleId,
      quoteType: qt,
      stockCode: code,
      stockName: name,
      conditionText: cond,
      enabled: true
    }

    // 7) 提前把 id 游标加 1：避免并发/重复点击时出现同 id
    this.nextRuleId += 1

    // 8) 更新状态：ArkUI 对数组引用敏感，所以这里用“新数组”触发 UI 刷新
    const next: NotifyRule[] = [newRule].concat(this.notifyRules)
    this.notifyRules = next
    this.notifyRulesVer += 1

    // 9) 持久化到 preferences：保证重启后规则还在
    await this.saveNotifyRulesToPrefs()

    // 10) 清空输入并给一个轻提示：让用户知道这次点击生效了
    this.ruleStockCodeInput = ''
    this.rulePriceInput = ''
    this.ruleMsg = '已添加并启用'
  }

  // 删除一条通知规则：按 id 过滤掉目标规则 → 更新列表版本号 → 立刻落盘到 preferences
  private async deleteRule(ruleId: number): Promise<void> {
    // 用新数组承接结果（不要原地 splice），这样 ArkUI 才能稳定触发刷新
    const next: NotifyRule[] = []

    // 遍历现有规则，把“不是要删的那条”都保留下来
    for (let i: number = 0; i < this.notifyRules.length; i++) {
      const r: NotifyRule = this.notifyRules[i]
      if (r.id !== ruleId) {
        next.push(r)
      }
    }

    // 替换为新数组：通知页面 List 会跟着重绘
    this.notifyRules = next

    // 手动 bump 一下版本号（你用它当 key），确保列表在某些情况下也会强制刷新
    this.notifyRulesVer += 1

    // 保存到本地：避免重启后“删掉的规则又回来了”
    await this.saveNotifyRulesToPrefs()
  }

  // 切换某张行情卡片的“展开 / 收起”状态（用 Map 记录，每张卡一个 key）
  private toggleCardExpanded(item: Quote): void {
    // key 里带上 type，避免指数/股票出现同码时互相覆盖（比如 000001）
    const key: string = `${item.type}-${item.code}`;

    // 旧状态：Map 里没存过就当作 false（默认收起）
    const oldVal: boolean = this.cardExpandedMap.get(key) ?? false;

    // 这里不直接改原来的 Map，而是复制一个新的：
    // ArkUI 的 @State 更容易识别“引用变了”，从而触发相关 UI 重绘
    const newMap: Map<string, boolean> = new Map<string, boolean>();
    this.cardExpandedMap.forEach((value: boolean, k: string) => {
      newMap.set(k, value);
    });

    // 把这张卡片的状态取反（展开 <-> 收起）
    newMap.set(key, !oldVal);

    // 替换为新 Map（通过“换引用”让状态更新生效）
    this.cardExpandedMap = newMap;
  }


  // ===== 自定义 TabBar：这里就能完全控制文字颜色 =====
  @Builder
  tabBarBuilder(title: string, index: number) {
    Column() {
      Text(title)
        .fontSize(14)
        .fontColor(this.currentTabIndex === index ? Color.White : '#888888')
    }
    .height(40)
    .width('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor(Color.Black)
  }

  // 页面主入口：上面是标题栏，下面用 Tabs 切三块（市场 / 自选 / 通知）
  build() {
    Column() {
      // 顶部标题栏（logo / 标题 / 右侧按钮之类的都在这里）
      this.buildHeader()

      // 三个 Tab：只负责“切页面”，具体内容分别交给 buildXXXTab()
      Tabs({ index: this.currentTabIndex }) {
        TabContent() {
          this.buildMarketTab()
        }
        .tabBar(this.tabBarBuilder('市场', 0)) // 市场行情：指数/股票列表 + K 线

        TabContent() {
          this.buildWatchlistTab()
        }
        .tabBar(this.tabBarBuilder('自选', 1)) // 自选分组：分组管理 + 自选列表 + 统计卡

        TabContent() {
          this.buildNotificationTab()
        }
        .tabBar(this.tabBarBuilder('通知', 2)) // 通知规则：规则增删改 + 命中后系统通知
      }
      .barMode(BarMode.Fixed)               // 固定 Tab 栏，不滚动
      .backgroundColor(Color.Black)
      .divider({ strokeWidth: 0.5, color: '#333333' }) // TabBar 下方那条细分割线
      .onChange((index: number) => {
        // 用户切换 Tab 时，更新当前索引（用于控制当前显示哪一页）
        this.currentTabIndex = index
      })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }

  // ====== Header 区域 ======
  @Builder
  private buildHeader() {
    Column() {
      Row({ space: 8 }) {
        Text('HARMONYSTOCK')
          .fontSize(22)
          .fontWeight(FontWeight.Bold)
          .fontColor(Color.White)

        Text('鸿股通')
          .fontSize(14)
          .fontColor('#AAAAAA')
          .margin({ top: 4 })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 16, bottom: 8 })
      .justifyContent(FlexAlign.SpaceBetween)
      .alignItems(VerticalAlign.Center)
    }
    .backgroundColor('#111111')
  }

  // ===== 模式切换：指数 / 股票 =====
  @Builder
  private buildModeSwitcher() {
    Row({ space: 24 }) {
      this.buildModeChip('指数', 'index');
      this.buildModeChip('股票', 'stock');
    }.width('100%')
    .justifyContent(FlexAlign.SpaceAround)
    .height(50)
  }

  // 市场页顶部的“指数 / 股票”切换按钮（一个 chip 对应一种模式）
  @Builder
  private buildModeChip(label: string, mode: 'index' | 'stock') {
    Text(label)
      .fontSize(16)
      // 当前模式选中态：文字高亮；未选中则灰色
      .fontColor(this.currentMode === mode ? '#FFFFFF' : '#AAAAAA')
      .padding({ left: 10, right: 10, top: 4, bottom: 4 })
      // 选中态给更亮的底色，做成“胶囊按钮”效果
      .backgroundColor(this.currentMode === mode ? '#333333' : '#181818')
      .borderRadius(12)
      .onClick(() => {
        // 点到已经选中的模式就不做任何事，避免重复请求/重复清状态
        if (this.currentMode === mode) {
          return;
        }

        // 切换行情模式（影响：列表接口 type、搜索 type、K 线 type）
        this.currentMode = mode;

        // 切模式时把“市场页相关状态”全部清掉，避免残留上一个模式的数据
        this.marketQuotes = [];
        this.searchedQuotes = [];
        this.selectedQuoteCode = '';
        this.selectedQuoteName = '';
        this.selectedQuoteType = mode;   // 当前选中标的的类型也跟着切
        this.selectedKlineIndex = -1;
        this.klinePoints = [];
        this.klineGeoms = [];
        this.klineXLabels = [];
        this.klineError = '';

        // 重新拉取该模式下的行情列表
        this.fetchQuotes(mode);
      })
  }


  // Tab 1：市场页（指数 / 股票切换 + 行情卡片横滑 + K 线展示）
  @Builder
  private buildMarketTab() {
    Column() {
      // 顶部模式切换（指数 / 股票两个 chip）
      this.buildModeSwitcher()

      // 顶部一行：左侧标题 + 右侧搜索框
      Row() {
        // 标题跟随 currentMode 变化，提示当前看的是什么列表
        Text(this.currentMode === 'index' ? '指数卡片' : '股票卡片')
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Color.White)
          .margin({ left: 16, top: 12, bottom: 4 })

        // 占位把右侧内容顶到最右边
        Blank()

        // 搜索区域：按 6 位代码查单个标的（指数/股票取决于 currentMode）
        this.buildSearchArea();
      }
      .width('100%')
      // 让标题、搜索框在垂直方向居中对齐（看起来更整齐）
      .alignItems(VerticalAlign.Center)

      // 行情横向滚动条：展示市场列表 + 搜索结果（可点选切换 K 线）
      this.buildQuoteStrip();

      // K 线卡片：展示当前选中标的的 K 线（周期切换、点击高亮等）
      this.buildKlineCard();
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    // Column 内部内容整体从左对齐（避免默认居中导致布局飘）
    .alignItems(HorizontalAlign.Start)
  }


  // 行情横滑 strip：上面优先展示“搜索结果”，后面是“后台实时列表”；
  // 同时把加载/错误状态也放在这里兜住，避免页面空白。
  @Builder
  private buildQuoteStrip() {
    // 行情列表请求中：给用户一个明确的加载反馈
    if (this.listLoading) {
      Row() {
        Text('行情加载中…')
          .fontSize(12)
          .fontColor('#888888')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })

      // 行情列表请求失败：展示错误信息（只影响当前模式的 UI）
    } else if (this.listError) {
      Row() {
        Text(`行情加载失败：${this.listError}`)
          .fontSize(12)
          .fontColor('#FF7875')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })

      // 正常状态：用横向 Scroll + Row 形成“卡片横滑”的效果
    } else {
      Scroll() {
        Row({ space: 12 }) {

          // ① 搜索出来的卡片：放最前面，相当于“置顶”，更方便你刚搜完马上点
          ForEach(this.searchedQuotes, (item: Quote) => {
            this.buildQuoteCard(item);
          }, (item: Quote) => `search-${item.type}-${item.code}`) // key：保证同一条数据稳定复用，不乱跳

          // ② 后台实时卡片：市场列表（轮询/刷新得到的）
          ForEach(this.marketQuotes, (item: Quote) => {
            this.buildQuoteCard(item);
          }, (item: Quote) => `rt-${item.type}-${item.code}`) // key：rt=real-time，避免和 search 的 key 冲突
        }
        // strip 的左右留白 + 上下内边距，卡片不会贴边
        .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      }
      // 关键：把 Scroll 设置成横向滚动，才能左右滑卡片
      .scrollable(ScrollDirection.Horizontal)
    }
  }


  // 搜索区：输入 6 位代码 + 点“查询”拉单只行情；
  // 下方顺带展示查询中的 loading / error，小而完整的一块 UI。
  @Builder
  private buildSearchArea() {
    Column() {

      // 第一行：输入框 + 查询按钮
      Row({ space: 8 }) {

        // 输入框：placeholder 会跟着当前模式切换（指数/股票），避免用户输错类型
        TextInput({
          placeholder: this.currentMode === 'index'
            ? '输入指数代码，例如 000001'
            : '输入股票代码，例如 300750'
        })
          .width('60%')
          .height(32)
          .fontSize(14)
          .backgroundColor(Color.White)
          .fontColor(Color.Black)
          // 用户每次输入变化，就把内容写进 searchCode（统一在 queryQuoteByCode 里校验长度）
          .onChange((value: string) => {
            this.searchCode = value.trim();
          })

        // 查询按钮：触发 queryQuoteByCode（里面会按 currentMode 去查 index 或 stock）
        Button('查询')
          .height(32)
          .onClick(() => {
            this.queryQuoteByCode();
          })
      }
      .padding({ left: 16, right: 16, top: 8, bottom: 4 })

      // 第二行：查询状态提示（只在需要时出现）
      if (this.searchLoading) {
        Row() {
          // 查询中给个轻提示，避免用户以为没点上
          Text('正在查询…')
            .fontSize(12)
            .fontColor('#888888')
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 8 })

      } else if (this.searchError) {
        Row() {
          // 查询失败：把错误信息直接露出来（例如“未找到该代码”“HTTP 500”等）
          Text(`查询失败：${this.searchError}`)
            .fontSize(12)
            .fontColor('#FF7875')
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 8 })
      }
    }
  }


  // ===== 行情卡片（指数 / 股票通用）=====
  // 这张卡做两件事：
  // 1) 点卡片主体：选中该标的 + 回写桌面卡片（如果是从桌面卡片进来的）+ 拉 K 线
  // 2) 点右下角 “＋/－”：只切换卡片展开/收起（不切换选中、不拉 K 线）
  @Builder
  private buildQuoteCard(item: Quote) {
    Column() {
      Column() {
        // ===== 卡片主体区域：显示名称/价格/涨跌 =====
        Column() {
          Row() {
            // 标的名称（指数名 / 股票名）
            Text(item.name)
              .fontSize(12)
              .fontColor('#AAAAAA')

            // 删除按钮：从 strip 里移除这张卡
            Text('✕')
              .fontSize(12)
              .fontColor('#777777')
              // 用 TapGesture 而不是 onClick：横向 Scroll 时更不容易被滑动手势抢走
              .parallelGesture(
                TapGesture().onAction(async () => {
                  await this.removeQuoteCard(item.code, item.type)
                })
              )
          }
          .width('100%')
          // 两端对齐：左边 name，右边 ✕
          .justifyContent(FlexAlign.SpaceBetween)
          // 垂直居中：让 name 和 ✕ 看起来在同一水平线上
          .alignItems(VerticalAlign.Center)

          // 当前价：展开时字体更大
          Text(item.price.toFixed(2))
          // 这里用 Map 里的状态控制“展开/收起”的字号
            .fontSize((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 28 : 20)
            .fontWeight(FontWeight.Bold)
            // 涨红跌绿（按你全局配色）
            .fontColor(item.change >= 0 ? '#FF4D4F' : '#39D98A')
            .margin({ top: 4 })

          // 涨跌额 + 涨跌幅：展开时字体更大
          Text(`${item.change.toFixed(2)}  ${item.changePercent.toFixed(2)}%`)
            .fontSize((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 16 : 12)
            .fontColor(item.change >= 0 ? '#FF7875' : '#6BE49B')
            .margin({ top: 4 })
        }

        // ===== 右下角展开/收起按钮：只改尺寸，不触发选中逻辑 =====
        Row() {
          Blank()
          Text((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? '－' : '＋')
            .fontSize(18)
            .fontColor('#CCCCCC')
            // 同样用 TapGesture，避免和外层“点卡片选中”的 tap 互相影响
            .parallelGesture(
              TapGesture().onAction(() => {
                this.toggleCardExpanded(item)
              })
            )
        }
        .width('100%')
        .margin({ top: 8 })
      }
      // 展开时 padding 更大，看起来“卡片更松”
      .padding((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 20 : 12)
      .backgroundColor('#171717')
      .borderRadius(12)
      // 展开时卡片更宽
      .width((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 200 : 140)
      // 展开时给一个固定高度；收起时不写 height，让它按内容自适应
      .height((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 150 : undefined)

      // ===== 卡片整体的点击：选中 + 回写桌面卡片 + 拉 K 线 =====
      // 用 parallelGesture 的原因：这张卡是在横向 Scroll 里，普通点击很容易被滚动手势吞掉
      .parallelGesture(
        TapGesture().onAction(async () => {
          // 这些日志是为了排查：是否点击事件没进来、activeFormId 是否为空等
          console.info(`[TAP] quoteCard tapped code=${item.code} activeFormId=${this.activeFormId}`)
          hilog.error(DOMAIN, TAG, '[TAP] quoteCard tapped code=%{public}s activeFormId=%{public}s',
            item.code, this.activeFormId)

          // 更新“当前选中标的”：K 线和右侧详细展示都靠这几个字段
          this.selectedQuoteCode = item.code
          this.selectedQuoteName = item.name
          this.selectedQuoteType = item.type
          hilog.error(DOMAIN, TAG, '[TAP] selected set type=%{public}s code=%{public}s', item.type, item.code)

          // 如果是从桌面卡片打开的（activeFormId 有值），就把当前选中的标的同步回桌面卡片
          hilog.error(DOMAIN, TAG, '[TAP] before updateWidgetIfLaunchedFromForm')
          await this.updateWidgetIfLaunchedFromForm(item)
          hilog.error(DOMAIN, TAG, '[TAP] after updateWidgetIfLaunchedFromForm')

          // 点击卡片后拉对应周期的 K 线
          const period: string = this.getKlinePeriodParam()
          this.fetchKline(item.type, item.code, period)
        })
      )
    }
  }


  // ===== K 线卡片 =====
  // 用一张卡把 “K线图 + 周期切换 + 当前蜡烛详情” 放在一起：上面是标题/周期，中间是图，下面是 x 轴日期。
  @Builder
  private buildKlineCard() {
    Column() {
      // 标题行：左边显示当前选中的标的名，右边是 日/周/月 切换
      Row({ space: 12 }) {
        Text(
          // 已经选中过某个行情卡片：优先用选中的名字做标题
          this.selectedQuoteName !== ''
            ? `${this.selectedQuoteName} K 线`
            // 否则就按当前模式给个默认标题
            : (this.currentMode === 'index' ? '指数 K 线' : '股票 K 线')
        )
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Color.White)

        Blank() // 把右侧的周期按钮顶到最右

        // 周期按钮：内部会切换 klinePeriodIndex 并触发重新拉取
        this.buildKlinePeriodChip('日K', 0)
        this.buildKlinePeriodChip('周K', 1)
        this.buildKlinePeriodChip('月K', 2)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 8 })

      // 当前选中的那根蜡烛的详情条（点某根 K 线后才会出现）
      if (this.selectedKlineIndex >= 0) {
        Row({ space: 8 }) {
          Text(this.selectedKlineDate) // 日期
            .fontSize(10)
            .fontColor('#CCCCCC')

          // OHLC：开/高/低/收
          Text(`开 ${this.selectedKlineOpen.toFixed(2)}`)
            .fontSize(10)
            .fontColor('#AAAAAA')

          Text(`高 ${this.selectedKlineHigh.toFixed(2)}`)
            .fontSize(10)
            .fontColor('#AAAAAA')

          Text(`低 ${this.selectedKlineLow.toFixed(2)}`)
            .fontSize(10)
            .fontColor('#AAAAAA')

          Text(`收 ${this.selectedKlineClose.toFixed(2)}`)
            .fontSize(10)
            .fontColor('#AAAAAA')
        }
        .width('100%')
        .padding({ left: 16, right: 16, top: 4, bottom: 4 })
      }

      // 主体内容：根据加载/错误/空数据/正常数据四种状态切换展示
      Column() {
        if (this.klineLoading) {
          Text('K 线加载中…')
            .fontSize(12)
            .fontColor('#888888')
        } else if (this.klineError) {
          Text(`K 线加载失败：${this.klineError}`)
            .fontSize(12)
            .fontColor('#FF7875')
        } else if (this.klineGeoms.length === 0) {
          Text('暂无 K 线数据')
            .fontSize(12)
            .fontColor('#666666')
        } else {
          // 有数据：左侧是 y 轴刻度，右侧是可横向滚动的蜡烛图 + x 轴标签
          Column() {
            Row() {
              // y 轴刻度：简单放 3 个（最大/中间/最小），用 SpaceBetween 撑开
              Column() {
                Text(this.klineMaxPrice.toFixed(0))
                  .fontSize(10)
                  .fontColor('#777777')
                Blank()
                Text(((this.klineMaxPrice + this.klineMinPrice) / 2).toFixed(0))
                  .fontSize(10)
                  .fontColor('#777777')
                Blank()
                Text(this.klineMinPrice.toFixed(0))
                  .fontSize(10)
                  .fontColor('#777777')
              }
              .width(40)
              .height(this.KLINE_PLOT_H)
              .justifyContent(FlexAlign.SpaceBetween)

              // 右侧图表：因为 K 线数量多，所以把“蜡烛区域 + x轴日期”一起放进横向 Scroll
              Scroll() {
                Column() {
                  // 蜡烛图区域：每根蜡烛用 KlineGeom 里预先算好的 topGap / wick / body 来拼出来
                  Row({ space: this.candleGap }) {
                    ForEach(this.klineGeoms, (g: KlineGeom, idx: number) => {
                      Column() {
                        // 顶部空白：把蜡烛整体“压”到正确的 y 位置
                        Blank().height(g.topGap)

                        // 上影线
                        Rect()
                          .width(2)
                          .height(g.upperWick)
                          .fill('#444444')

                        // 实体：涨跌用不同颜色（注意：这里的 isUp 定义是 close >= open）
                        Rect()
                          .width(this.candleBodyWidth)
                          .height(g.bodyHeight)
                          .fill(g.isUp ? '#39D98A' : '#FF4D4F')

                        // 下影线
                        Rect()
                          .width(2)
                          .height(g.lowerWick)
                          .fill('#444444')

                        Blank().height(4)
                      }
                      // 高亮当前选中的蜡烛：用边框做一个轻量的“选中框”
                      .border({
                        width: this.selectedKlineIndex === idx ? 1 : 0,
                        color: this.selectedKlineIndex === idx ? '#FACC15' : '#000000'
                      })
                      // 点某根蜡烛：更新 selectedKlineIndex 和详情条展示的数据
                      .parallelGesture(
                        TapGesture().onAction(() => {
                          this.onKlineSelected(idx)
                        })
                      )
                    }, (g: KlineGeom, idx: number) => `${idx}`)
                  }
                  // 总宽度 = 蜡烛数量 * (实体宽 + 间距) 之类的逻辑（你在 getKlineTotalWidth() 里算）
                  .width(this.getKlineTotalWidth())
                  .height(this.KLINE_PLOT_H)

                  // x 轴日期标签：用 SpaceBetween 把少量标签均匀铺开
                  if (this.klineXLabels.length > 0) {
                    Row() {
                      ForEach(this.klineXLabels, (label: string, idx: number) => {
                        Text(label)
                          .fontSize(10)
                          .fontColor('#777777')
                      }, (label: string, idx: number) => `${idx}-${label}`)
                    }
                    .width(this.getKlineTotalWidth())
                    .justifyContent(FlexAlign.SpaceBetween)
                    .margin({ bottom: 10 })
                  }
                }
                // 右侧留一点边距，避免最后一根蜡烛贴边不好点
                .padding({ right: 32 })
              }
              .scrollable(ScrollDirection.Horizontal)
              // 图 + x轴一起滚，所以高度是 plot + label
              .height(this.KLINE_PLOT_H + this.KLINE_LABEL_H)
              .margin({ left: 8 })
            }
            .width('100%')
          }
        }
      }
      // 卡片整体的固定高度：给内容留出空间（图 + x轴 + 标题/详情条的余量）
      .width('100%')
      .height(this.KLINE_PLOT_H + this.KLINE_LABEL_H + 70)
      .margin({ top: 8, bottom: 12 })
      .padding(16)
      .backgroundColor('#101010')
      .borderRadius(16)
    }
  }

  // K 线周期切换按钮：点“日/周/月”时，更新 klinePeriodIndex，并按当前选中的标的重新拉一次 K 线
  @Builder
  private buildKlinePeriodChip(label: string, index: number) {
    Text(label)
      .fontSize(12)
      // 选中态高亮：当前周期 == 这个 chip 的 index
      .fontColor(this.klinePeriodIndex === index ? '#FFFFFF' : '#AAAAAA')
      .padding({ left: 10, right: 10, top: 4, bottom: 4 })
      .backgroundColor(this.klinePeriodIndex === index ? '#333333' : '#181818')
      .borderRadius(12)
      .onClick(() => {
        // 点到同一个周期就不做事，避免重复请求 + 无意义刷新
        if (this.klinePeriodIndex === index) {
          return;
        }

        // 先更新周期索引（getKlinePeriodParam() 会根据它返回 day/week/month）
        this.klinePeriodIndex = index;

        // 只有“已经选中过某个行情卡片”才拉 K 线
        // （没选中的话 selectedQuoteCode 为空，拉了也不知道拉谁）
        if (this.selectedQuoteCode !== '') {
          const period: string = this.getKlinePeriodParam();
          // 用当前选中的类型 + 代码重新拉一份 K 线数据
          this.fetchKline(this.selectedQuoteType, this.selectedQuoteCode, period);
        }
      })
  }


  // Tab 2（自选）：分组管理 + 当前分组的股票列表（依赖 activeGroupId / activeStocks / quoteCache 来驱动 UI）
  @Builder
  private buildWatchlistTab() {
    Column() {
      // 顶部：分组横向条（切换分组、删除分组等一般都在这里做）
      this.buildGroupStrip()

      // 新建分组输入栏
      this.buildCreateGroupBar()

      // 页面级提示信息（比如“已添加 xx 只”“已删除 xx 只”之类）
      if (this.watchlistMsg.length > 0) {
        Text(this.watchlistMsg)
          .fontSize(10)
          .fontColor('#777777')
          .margin({ left: 16, bottom: 8 })
      }

      // 没有任何分组时的空态
      if (this.activeGroupId < 0 || this.groups.length === 0) {
        Text('暂无分组，请先创建一个分组')
          .fontSize(12)
          .fontColor('#777777')
          .margin({ top: 24 })
          .width('100%')
          .textAlign(TextAlign.Center)
      } else {
        // 当前分组标题区（直接读 activeGroupId 对应的 group 信息）
        this.buildActiveGroupHeaderCurrent()

        // 当前分组统计卡（股票数 / 涨跌数 / 平均涨跌幅，依赖 quoteCache）
        this.buildGroupStatsCard()

        // 组内添加股票输入栏
        this.buildAddStockBar()

        // 当前分组没有股票时的空态
        if (this.getActiveGroupStocksList().length === 0) {
          Text('该分组暂无股票，请在下方输入代码添加')
            .fontSize(12)
            .fontColor('#666666')
            .margin({ top: 12, left: 16 })
        } else {
          // 组内股票列表：用 activeStocks（从 activeGroup.stocks slice 出来的“新数组”）
          List() {
            ForEach(this.activeStocks, (s: StockRef) => {
              this.buildStockRefRow(this.activeGroupId, s)
            }, (s: StockRef) => {
              // 行级 key：保证同一分组内 code 唯一即可
              return `${this.activeGroupId}-${s.code}`
            })
          }
          // List 级 key：activeStocksVer 变化时强制 List 重建，解决“数组内容变了但列表不刷新”的情况
          .key(`${this.activeGroupId}-${this.activeStocksVer}`)
          .padding({ left: 16, right: 16, top: 8, bottom: 12 })
        }
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    // 页面级 key：把几个关键版本号都带上，保证分组切换/行情更新/统计刷新时，这个 Tab 的关键区域能稳定重绘
    .key(`watchlist-${this.activeGroupId}-${this.activeStocksVer}-${this.quoteCacheVer}-${this.activeStatsVer}`)
  }


  // 当前分组的“标题行”：左侧显示分组名，右侧提供“删除分组”入口（直接操作 activeGroupId）
  @Builder
  private buildActiveGroupHeaderCurrent() {
    Row({ space: 8 }) {
      // 分组名称：从 activeGroupId 对应的 group 里取 name（空时返回 ''）
      Text(this.getActiveGroupNameText())
        .fontSize(14)
        .fontColor('#FFFFFF')
        .fontWeight(FontWeight.Medium)

      Blank()

      // 删除按钮：删除当前 activeGroupId 对应的分组，并触发 active 链路同步（你在 deleteGroup 里已经做了）
      Text('删除分组')
        .fontSize(12)
        .fontColor('#FF7875')
        .padding({ left: 10, right: 10, top: 6, bottom: 6 })
        .backgroundColor('#1A1A1A')
        .borderRadius(10)
        .onClick(() => {
          this.deleteGroup(this.activeGroupId)
        })
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 8, bottom: 6 })
  }


  // 分组横向滚动条：展示所有分组，并允许点击切换当前 active 分组
  @Builder
  private buildGroupStrip() {
    Scroll() {
      Row({ space: 8 }) {
        ForEach(this.groups, (group: StockGroup) => {
          Column({ space: 2 }) {
            // 分组名称
            Text(group.name)
              .fontSize(12)
              // 当前选中的分组高亮显示
              .fontColor(this.activeGroupId === group.id ? '#FFFFFF' : '#DDDDDD')

            // 分组平均涨跌幅（来自缓存行情的统计结果）
            Text(this.getGroupAvgPctText(group.id))
              .fontSize(10)
              .fontColor(this.getGroupAvgPctColor(group.id))
          }
          // 每个分组 chip 的整体样式
          .padding({ left: 12, right: 12, top: 6, bottom: 6 })
          .backgroundColor(this.activeGroupId === group.id ? '#2A2A2A' : '#161616')
          .borderRadius(12)
          .onClick(async () => {
            // 点击切换当前分组
            this.activeGroupId = group.id

            // 先同步一次 activeStocks，保证列表立即切换
            this.syncActiveStocks()

            // 预热该分组下所有股票的行情（补齐缓存）
            await this.preloadActiveGroupQuotes()

            // 行情回来后再同步一次，确保统计和列表都用到最新数据
            this.syncActiveStocks()
          })
        }, (group: StockGroup) => group.id.toString())
      }
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })
    }
    // 横向滚动，用来容纳较多分组
    .scrollable(ScrollDirection.Horizontal)
  }


  // 新建分组输入栏：输入分组名并创建一个新的自选分组
  @Builder
  private buildCreateGroupBar() {
    Row({ space: 8 }) {
      // 分组名输入框（受控输入，值直接绑定到 newGroupName）
      TextInput({
        placeholder: '新建分组名（如 科技股）',
        text: this.newGroupName
      })
        .width('68%')
        .height(32)
        .fontSize(13)
        .backgroundColor(Color.White)
        .fontColor(Color.Black)
        .onChange((v: string) => {
          // 实时更新 state，后面点击“创建”时直接用
          this.newGroupName = v.trim();
        })

      // 创建按钮：校验输入并调用 createGroup
      Button('创建')
        .height(32)
        .onClick(async () => {
          const name: string = this.newGroupName.trim();
          if (name.length === 0) {
            // 简单的输入校验提示
            this.watchlistMsg = '请输入分组名';
            return;
          }

          // 真正创建分组（内部会更新 groups / activeGroupId / 持久化）
          await this.createGroup(name);
        })
    }
    .padding({ left: 16, right: 16, bottom: 8 })
  }


  // 分组统计卡：展示当前 active 分组的整体情况（股票数 / 上涨 / 下跌 / 平均涨跌幅）
  @Builder
  private buildGroupStatsCard() {
    Column() {
      Row({ space: 12 }) {

        // 股票总数
        Column({ space: 4 }) {
          Text('股票数')
            .fontSize(10)
            .fontColor('#777777')
          Text(this.activeStatStockCount)
            .fontSize(14)
            .fontColor('#FFFFFF')
            .fontWeight(FontWeight.Medium)
        }.width('22%')

        // 上涨数量
        Column({ space: 4 }) {
          Text('上涨')
            .fontSize(10)
            .fontColor('#777777')
          Text(this.activeStatUp)
            .fontSize(14)
            .fontColor('#FFFFFF')
            .fontWeight(FontWeight.Medium)
        }.width('22%')

        // 下跌数量
        Column({ space: 4 }) {
          Text('下跌')
            .fontSize(10)
            .fontColor('#777777')
          Text(this.activeStatDown)
            .fontSize(14)
            .fontColor('#FFFFFF')
            .fontWeight(FontWeight.Medium)
        }.width('22%')

        // 平均涨跌幅（可能是 '--'，表示当前还没有有效行情）
        Column({ space: 4 }) {
          Text('均涨跌')
            .fontSize(10)
            .fontColor('#777777')
          Text(this.activeStatAvgPct)
            .fontSize(14)
            .fontColor('#FFFFFF')
            .fontWeight(FontWeight.Medium)
        }.width('22%')
      }
      .width('100%')
      .padding(12)
      .margin({ left: 16, right: 16, bottom: 8 })
      .backgroundColor('#101010')
      .borderRadius(14)
    }
    // key 里带上分组 / 行情 / 统计版本号，确保相关数据变化时这张统计卡会重新渲染
    .key(`stats-${this.activeGroupId}-${this.activeStocksVer}-${this.quoteCacheVer}-${this.activeStatsVer}`)
  }

  // 添加股票输入栏：向当前 active 分组里追加股票代码
  @Builder
  private buildAddStockBar() {
    Row({ space: 8 }) {
      // 股票代码输入框（受控输入，绑定 addStockCodeInput）
      TextInput({
        placeholder: '输入 6 位股票代码（如 300750）',
        text: this.addStockCodeInput
      })
        .width('68%')
        .height(32)
        .fontSize(13)
        .backgroundColor(Color.White)
        .fontColor(Color.Black)
        .onChange((v: string) => {
          // 实时同步输入值，点击“添加”时直接使用
          this.addStockCodeInput = v.trim();
        })

      // 添加按钮：校验输入并把股票加到当前分组
      Button('添加')
        .height(32)
        .onClick(async () => {
          const code: string = this.addStockCodeInput.trim();

          // 基本校验：必须是 6 位代码
          if (code.length !== 6) {
            this.watchlistMsg = '请输入 6 位代码';
            return;
          }

          // 永远以当前 activeGroupId 为准，避免 Builder 闭包里拿到旧值
          const gid: number = this.activeGroupId;
          if (gid < 0) {
            this.watchlistMsg = '当前没有可用分组';
            return;
          }

          // 调用分组逻辑添加股票（内部会去重、持久化、刷新统计）
          await this.addStocksToGroup(gid, [code]);

          // 简单的操作反馈提示
          this.watchlistMsg = `已尝试添加：${code} -> group=${gid}`;
        })
    }
    .padding({ left: 16, right: 16, bottom: 8 })
  }

  // 自选列表中的单行股票项：展示名称 / 代码 / 行情信息，并提供删除入口
  @Builder
  private buildStockRefRow(groupId: number, s: StockRef) {
    ListItem() {
      Row() {
        // 左侧：股票名称 + 代码
        Column({ space: 4 }) {
          // 名称优先从行情缓存里取，兜底用 StockRef 里的 name / 占位符
          Text(this.getStockRowName(s))
            .fontSize(14)
            .fontColor(Color.White)

          // 股票代码（始终显示）
          Text(s.code)
            .fontSize(11)
            .fontColor('#666666')
        }

        Blank() // 撑开左右两列

        // 右侧：价格 + 涨跌信息（右对齐，更像行情列表）
        Column({ space: 4 }) {
          // 当前价格，颜色按涨跌区分
          Text(this.getStockRowPriceText(s))
            .fontSize(14)
            .fontColor(this.getStockRowPriceColor(s))
            .textAlign(TextAlign.End)

          // 涨跌额 + 涨跌幅（次要信息，用浅一点的颜色）
          Text(this.getStockRowChangeText(s) + '  ' + this.getStockRowPctText(s))
            .fontSize(11)
            .fontColor(this.getStockRowSubColor(s))
            .textAlign(TextAlign.End)
        }

        // 删除按钮：把当前股票从该分组中移除
        Button('✕')
          .height(28)
          .width(28)
          .padding(0)
          .backgroundColor('#1A1A1A')
          .fontColor('#777777')
          .borderRadius(14)
          .onClick(() => {
            // 这里只做同步删除调用，不用 async，避免手势链路变复杂
            this.removeStocksFromGroup(groupId, [s.code])
          })
      }
      .padding({ left: 12, right: 12, top: 10, bottom: 10 })
      .backgroundColor('#101010')
      .borderRadius(12)
      .width('100%')
    }
    // 行与行之间留一点间距
    .margin({ bottom: 8 })
    .width('100%')
  }


  // ====== Tab 3：行情通知中心（只负责“规则的增删展示”，真正的触发在 RuleEngine） ======
  @Builder
  private buildNotificationTab() {
    Column() {
      // 顶部标题栏：这里只做标题展示，不放复杂交互
      Row() {
        Text('通知规则').fontSize(14).fontColor(Color.White)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })

      // 新增规则区：把“类型 + 代码 + 条件 + 添加按钮”集中放在一起
      Column({ space: 8 }) {
        Column({ space: 8 }) {
          // 规则类型：股票 / 指数（二选一，影响 placeholder 和后端查询 type）
          Row({ space: 6 }) {
            this.buildRuleTypeChip('股票', 'stock')
            this.buildRuleTypeChip('指数', 'index')
          }

          Row({ space: 8 }) {
            // 代码输入：6 位；placeholder 会跟随当前 ruleQuoteType 切换
            TextInput({
              placeholder: this.ruleQuoteType === 'stock' ? '股票代码(6位)' : '指数代码(6位)',
              text: this.ruleStockCodeInput
            })
              .width('28%')
              .height(32)
              .fontSize(13)
              .backgroundColor(Color.White)
              .fontColor(Color.Black)
              // 输入时把两边空格去掉，避免“看着一样其实多了空格”导致校验失败
              .onChange((v: string) => { this.ruleStockCodeInput = v.trim() })

            // 操作符选择：用两个按钮（≥ / ≤），避免 Picker 的手势冲突和类型麻烦
            Row({ space: 6 }) {
              Text('≥')
                .fontSize(12)
                .fontColor(this.ruleOpIndex === 0 ? '#FFFFFF' : '#AAAAAA')
                .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                .backgroundColor(this.ruleOpIndex === 0 ? '#333333' : '#181818')
                .borderRadius(10)
                .onClick(() => { this.ruleOpIndex = 0 })

              Text('≤')
                .fontSize(12)
                .fontColor(this.ruleOpIndex === 1 ? '#FFFFFF' : '#AAAAAA')
                .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                .backgroundColor(this.ruleOpIndex === 1 ? '#333333' : '#181818')
                .borderRadius(10)
                .onClick(() => { this.ruleOpIndex = 1 })
            }

            // 阈值输入：比如 12.34；最终会拼进 conditionText（"价格 ≥ xx.xx"）
            TextInput({ placeholder: '阈值', text: this.rulePriceInput })
              .width('18%')
              .height(32)
              .fontSize(13)
              .backgroundColor(Color.White)
              .fontColor(Color.Black)
              .onChange((v: string) => { this.rulePriceInput = v.trim() })

            // 添加按钮：真正的校验 / 去重 / 写入 prefs 都在 addNotifyRuleFromUI 里做
            Button('添加')
              .height(32)
              .onClick(async () => { await this.addNotifyRuleFromUI() })
          }
        }

        // UI 提示：比如“请输入 6 位代码 / 已添加并启用”
        if (this.ruleMsg.length > 0) {
          Text(this.ruleMsg)
            .fontSize(10)
            .fontColor('#777777')
        }
      }
      .padding({ left: 16, right: 16, bottom: 10 })

      // 规则列表：只是把 notifyRules 渲染成可操作的行（启用/删除等）
      List() {
        ForEach(this.notifyRules, (rule: NotifyRule) => {
          this.buildNotifyRuleItem(rule)
        }, (rule: NotifyRule) => rule.id.toString())
      }
      // 用版本号做 key：确保增删规则后 List 能重新创建，避免偶发不刷新
      .key(`notifyRules-${this.notifyRulesVer}`)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }


  // 单条通知规则的列表行：展示“类型 + 名称/代码 + 条件”，并提供删除入口
  @Builder
  private buildNotifyRuleItem(rule: NotifyRule) {
    ListItem() {
      Row({ space: 10 }) {
        Column({ space: 4 }) {
          Row({ space: 6 }) {
            // 类型小标签：让用户一眼能区分这条规则是股票还是指数
            Text(rule.quoteType === 'stock' ? '股票' : '指数')
              .fontSize(10)
              .fontColor('#AAAAAA')
              .padding({ left: 6, right: 6, top: 2, bottom: 2 })
              .backgroundColor('#1A1A1A')
              .borderRadius(6)

            // 主标题：标的名称 + 代码（跟行情卡片的展示习惯保持一致）
            Text(`${rule.stockName}（${rule.stockCode}）`)
              .fontSize(13)
              .fontColor(Color.White)
          }

          // 条件文本：直接复用你保存的 conditionText（例如 “价格 ≥ 12.34”）
          Text(rule.conditionText)
            .fontSize(11)
            .fontColor('#BBBBBB')
        }
        .layoutWeight(1) // 左侧信息区占满剩余空间，把“删除”按钮顶到右边

        // 删除按钮：只负责触发删除逻辑，具体持久化在 deleteRule 里做
        Button('删除')
          .height(28)
          .padding({ left: 10, right: 10 })
          .backgroundColor('#1A1A1A')
          .fontColor('#FF7875')
          .borderRadius(10)
          .onClick(async () => {
            await this.deleteRule(rule.id)
          })
      }
      .padding({ left: 16, right: 16, top: 10, bottom: 10 })
      .backgroundColor('#101010')
    }
    .margin({ bottom: 1 }) // 行与行之间留一点缝，视觉上更像“列表”
  }


  // 规则类型选择按钮：用于在“股票 / 指数”之间切换当前要创建的通知规则类型
  @Builder
  private buildRuleTypeChip(label: string, t: 'index' | 'stock') {
    Text(label)
      .fontSize(12)
      // 当前选中的类型高亮显示，其余置灰
      .fontColor(this.ruleQuoteType === t ? '#FFFFFF' : '#AAAAAA')
      .padding({ left: 10, right: 10, top: 6, bottom: 6 })
      .backgroundColor(this.ruleQuoteType === t ? '#333333' : '#181818')
      .borderRadius(10)
      .onClick(() => {
        // 点击后直接切换当前规则的 quoteType
        // 后续输入框 placeholder、查询类型都会随之变化
        this.ruleQuoteType = t
      })
  }

}
