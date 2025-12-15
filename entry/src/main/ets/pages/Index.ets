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
    const code6: string = code.trim();
    if (code6.length !== 6) return null;

    const kStock: string = `stock-${code6}`;
    const vStock: Quote | undefined = this.quoteCache.get(kStock);
    if (vStock !== undefined) return vStock;

    const kIndex: string = `index-${code6}`;
    const vIndex: Quote | undefined = this.quoteCache.get(kIndex);
    if (vIndex !== undefined) return vIndex;

    return null;
  }

  private getStockRowName(s: StockRef): string {
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    if (q !== null) {
      return q.name;
    }
    if (s.name && s.name.length > 0) {
      return s.name;
    }
    return '—';
  }

  private getStockRowPriceText(s: StockRef): string {
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    return (q !== null) ? q.price.toFixed(2) : '--';
  }

  private getStockRowChangeText(s: StockRef): string {
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    return (q !== null) ? q.change.toFixed(2) : '--';
  }

  private getStockRowPctText(s: StockRef): string {
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    return (q !== null) ? (q.changePercent.toFixed(2) + '%') : '--';
  }

  private getStockRowPriceColor(s: StockRef): string {
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    if (q === null) {
      return '#AAAAAA';
    }
    return (q.change >= 0) ? '#FF4D4F' : '#39D98A';
  }

  private getStockRowSubColor(s: StockRef): string {
    const _qv: number = this.quoteCacheVer
    const q: Quote | null = this.findStockQuoteInCache(s.code);

    if (q === null) {
      return '#777777';
    }
    return (q.change >= 0) ? '#FF7875' : '#6BE49B';
  }

  private async ensureQuoteCachedForCode(code: string): Promise<void> {
    const code6: string = code.trim()
    if (code6.length !== 6) return

    // ✅ 仅检查 stock 缓存，避免 index 同码导致误判
    const cacheKey: string = `stock-${code6}`
    const existed: Quote | undefined = this.quoteCache.get(cacheKey)
    if (existed !== undefined) return

    const httpRequest = http.createHttp()
    try {
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
        this.mergeQuotesToCache([q]) // ✅ quoteCacheVer++ 会触发统计卡刷新
        this.dbg(`ensureQuoteCachedForCode:request ${code6}`)
      }
    } catch (e) {
      // ignore
    } finally {
      httpRequest.destroy()
    }
  }

  private async ensureQuoteCachedByType(type: QuoteType, code6: string): Promise<void> {
    const code: string = code6.trim()
    if (code.length !== 6) return

    const cacheKey: string = `${type}-${code}`
    const existed: Quote | undefined = this.quoteCache.get(cacheKey)
    if (existed !== undefined) return

    const httpRequest = http.createHttp()
    try {
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`
      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })
      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        const q: Quote = this.mapBackendQuote(backendItem, type)
        this.mergeQuotesToCache([q])
      }
    } catch (e) {
      // ignore
    } finally {
      httpRequest.destroy()
    }
  }

  private findQuoteInCacheByType(type: QuoteType, code6: string): Quote | null {
    const code: string = code6.trim()
    if (code.length !== 6) return null
    const k: string = `${type}-${code}`
    const v: Quote | undefined = this.quoteCache.get(k)
    return (v !== undefined) ? v : null
  }

  private async preloadActiveGroupQuotes(): Promise<void> {
    const g: StockGroup | null = this.getActiveGroup();
    if (g === null) {
      return;
    }

    const tasks: Promise<void>[] = [];
    for (let i: number = 0; i < g.stocks.length; i++) {
      const code: string = g.stocks[i].code;
      tasks.push(this.ensureQuoteCachedForCode(code));
    }

    // 并行预热：单只失败不会阻塞整体
    await Promise.all(tasks.map((p: Promise<void>) => p.catch(() => { })));

  }

  private mergeQuotesToCache(list: Quote[]): void {
    console.info(`[DBG] mergeQuotesToCache: in list=${list.length} oldCache=${this.quoteCache.size}`);
    const newMap: Map<string, Quote> = new Map<string, Quote>();
    this.quoteCache.forEach((v: Quote, k: string) => {
      newMap.set(k, v);
    });

    for (let i: number = 0; i < list.length; i++) {
      const q: Quote = list[i];
      const key: string = `${q.type}-${q.code}`;
      newMap.set(key, q);
    }

    this.quoteCache = newMap;
    this.quoteCacheVer += 1
    this.refreshActiveGroupStats()   // ✅ 新增：行情变了，统计（up/down/avg）立刻刷新
    console.info(`[DBG] mergeQuotesToCache: out newCache=${this.quoteCache.size}`);
  }


  private getActiveGroupOrNull(): StockGroup | null {
    if (this.activeGroupId < 0) return null
    const idx: number = this.findGroupIndex(this.activeGroupId)
    if (idx < 0) return null
    return this.groups[idx]
  }

  private getActiveGroupNameText(): string {
    const g: StockGroup | null = this.getActiveGroupOrNull()
    return g ? g.name : ''
  }

  private getActiveGroupStocksList(): StockRef[] {
    const g: StockGroup | null = this.getActiveGroupOrNull()
    return g ? g.stocks : []
  }

  private computeStatsFromStocks(list: StockRef[]): ActiveGroupStats {
    const stockCount: number = list.length

    let up: number = 0
    let down: number = 0
    let sumPct: number = 0
    let cntPct: number = 0

    for (let i: number = 0; i < list.length; i++) {
      const code: string = list[i].code
      // const q: Quote | null = this.findQuoteInCaches(code)
      const q: Quote | null = this.findStockQuoteInCache(code)
      if (q !== null) {
        if (q.change >= 0) {
          up += 1
        } else {
          down += 1
        }
        sumPct += q.changePercent
        cntPct += 1
      }
    }

    const avgPct: string = (cntPct > 0) ? ((sumPct / cntPct).toFixed(2) + '%') : '--'

    return {
      stockCount: stockCount.toString(),
      up: up.toString(),
      down: down.toString(),
      avgPct: avgPct
    }
  }

  private getActiveGroupStatsStockCount(): string {
    const _av: number = this.activeStocksVer     // ✅ 让“增删股票”触发刷新
    const _qv: number = this.quoteCacheVer       // ✅ 让“行情回写”触发刷新
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.activeStocks)
    return st.stockCount
  }

  private getActiveGroupStatsUp(): string {
    const _av: number = this.activeStocksVer
    const _qv: number = this.quoteCacheVer
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.activeStocks)
    return st.up
  }

  private getActiveGroupStatsDown(): string {
    const _av: number = this.activeStocksVer
    const _qv: number = this.quoteCacheVer
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.activeStocks)
    return st.down
  }

  private getActiveGroupStatsAvgPct(): string {
    const _av: number = this.activeStocksVer
    const _qv: number = this.quoteCacheVer
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.activeStocks)
    return st.avgPct
  }

  // ✅ 自选分组专用：只按 stock 查缓存，避免与 index 同码冲突（如 000001）
  private findStockQuoteInCache(code: string): Quote | null {
    const code6: string = code.trim()
    if (code6.length !== 6) return null

    const kStock: string = `stock-${code6}`
    const v: Quote | undefined = this.quoteCache.get(kStock)
    return (v !== undefined) ? v : null
  }



  private logSeq: number = 0;

  private dbg(tag: string): void {
    this.logSeq += 1;

    const gid: number = this.activeGroupId;
    const idx: number = this.findGroupIndex(gid);

    let activeCount: number = -1;
    let activeName: string = '';
    if (idx >= 0) {
      activeCount = this.groups[idx].stocks.length;
      activeName = this.groups[idx].name;
    }

    console.info(
      `[DBG#${this.logSeq}] ${tag} | tab=${this.currentTabIndex} mode=${this.currentMode}` +
        ` | activeGroupId=${gid} activeIdx=${idx} activeName=${activeName} activeStocks=${activeCount}` +
        ` | groups=${this.groups.length} cache=${this.quoteCache.size}` +
        ` | activeStocks=${this.activeStocks.length} cache=${this.quoteCache.size} cacheVer=${this.quoteCacheVer}`
    );
  }

  private syncActiveStocks(): void {
    const g: StockGroup | null = this.getActiveGroupOrNull()
    if (g === null) {
      this.activeStocks = []
    } else {
      // 关键：必须创建“新数组引用”
      this.activeStocks = g.stocks.slice()
    }
    this.activeStocksVer += 1

    this.refreshActiveGroupStats()
  }

  private getGroupAvgPctText(groupId: number): string {
    // ✅ 依赖缓存版本：行情更新 => quoteCacheVer++ => 这里会重新计算 => UI 重绘
    const _qv: number = this.quoteCacheVer

    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) return '--'

    const st: ActiveGroupStats = this.computeStatsFromStocks(this.groups[idx].stocks)
    return st.avgPct
  }

  private getGroupAvgPctColor(groupId: number): string {
    // ✅ 依赖行情缓存版本，保证行情变化时颜色也刷新
    const _qv: number = this.quoteCacheVer

    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) return '#777777'

    const st: ActiveGroupStats = this.computeStatsFromStocks(this.groups[idx].stocks)
    const avg: string = st.avgPct

    // 没有行情
    if (avg === '--') {
      return '#777777'
    }

    // 去掉 %
    const v: number = Number(avg.replace('%', ''))
    if (Number.isNaN(v)) {
      return '#777777'
    }

    // ✅ 涨红，跌绿（按你当前全局配色）
    return v >= 0 ? '#FF4D4F' : '#39D98A'
  }

  // ✅ 方案1第三步：把 Quote 转成桌面卡片绑定数据
  private buildWidgetDataFromQuote(q: Quote): WidgetBindData {
    const now: Date = new Date()
    const hh: string = now.getHours().toString().padStart(2, '0')
    const mm: string = now.getMinutes().toString().padStart(2, '0')

    return {
      formId: this.activeFormId,
      title: (q.type === 'index') ? '指数' : '股票',
      code: q.code,
      name: q.name,
      priceText: q.price.toFixed(2),
      changeText: q.change.toFixed(2),
      pctText: q.changePercent.toFixed(2) + '%',
      tsText: `${hh}:${mm}`,
      hasPrev: true,
      hasNext: true
    }
  }

  // ✅ 方案1第三步：如果是从桌面卡片打开的 App（activeFormId 有值），就把当前选择回写到卡片
  private async updateWidgetIfLaunchedFromForm(q: Quote): Promise<void> {
    const fid: string = this.activeFormId.trim()

    hilog.error(DOMAIN, TAG, '[WIDGET] ENTER updateWidget fid=%{public}s code=%{public}s', fid, q.code)

    if (fid.length === 0) {
      hilog.error(DOMAIN, TAG, '[WIDGET] SKIP activeFormId empty')
      return
    }

    try {
      const data: WidgetBindData = this.buildWidgetDataFromQuote(q)
      const binding: formBindingData.FormBindingData = formBindingData.createFormBindingData(data)

      await formProvider.updateForm(fid, binding)

      hilog.error(DOMAIN, TAG, '[WIDGET] OK updateForm formId=%{public}s code=%{public}s', fid, q.code)
    } catch (e) {
      hilog.error(DOMAIN, TAG, '[WIDGET] FAIL updateForm formId=%{public}s err=%{public}s',
        fid, JSON.stringify(e))
    }
  }



  private async triggerRuleAlert(rule: NotifyRule): Promise<void> {
    try {
      // 先尝试拉起通知授权弹窗（若已开启，通常不会影响；若未开启会提示用户开启）
      // 若用户拒绝，常见错误码为 1600004
      try {
        await notificationManager.requestEnableNotification()
      } catch (e) {
        // 不使用 any/unknown：只做最保守处理
        const errObj: Record<string, Object> = e as Record<string, Object>
        const codeNum: number = Number(errObj['code'])
        hilog.error(DOMAIN, TAG, 'requestEnableNotification failed: %{public}s', JSON.stringify(e))

        // 1600004：用户拒绝/通知未授权（或已被关闭）
        if (codeNum === 1600004) {
          // 这里你可以改成 UI 提示：引导用户到系统设置开启通知
          return
        }
      }

      const req: notificationManager.NotificationRequest = {
        id: 10000 + rule.id,
        content: {
          // 关键：用 notificationContentType（不是 contentType）
          notificationContentType: notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
          normal: {
            title: '行情提醒',
            text: `${rule.stockName}（${rule.stockCode}）${rule.conditionText}`,
            additionalText: '打开应用查看详情'
          }
        }
        // slotType / wantAgent / autoDeleted：先都不写，避免你当前 SDK 枚举/类型不一致导致编译或运行失败
      }

      await notificationManager.publish(req)
      hilog.info(DOMAIN, TAG, 'triggerRuleAlert ok, id=%{public}d', req.id)
    } catch (e) {
      hilog.error(DOMAIN, TAG, 'triggerRuleAlert failed: %{public}s', JSON.stringify(e))
    }
  }



  // ✅ 先检查通知是否开启；没开就拉起系统授权/开关页面
  private async ensureNotificationEnabled(): Promise<boolean> {
    try {
      const enabled: boolean = await notificationManager.isNotificationEnabled()
      if (enabled) {
        return true
      }

      // 拉起系统界面让用户打开通知开关
      await notificationManager.requestEnableNotification()

      // 用户操作后再检查一次
      const enabled2: boolean = await notificationManager.isNotificationEnabled()
      return enabled2
    } catch (e) {
      // 任意异常都视为不可用
      return false
    }
  }

  // ✅ 专门处理 publish 的 1600004：提示 + 引导开启
  private async handlePublishError(code: number): Promise<void> {
    if (code === 1600004) {
      // 你可以用你现有的 UI 状态字段提示用户
      this.watchlistMsg = '通知权限未开启：请在系统设置中为 HarmonyStock 打开通知后重试'
      // 再主动拉起一次系统开关界面
      await this.ensureNotificationEnabled()
    }
  }


  private async loadNotifyRulesFromPrefs(): Promise<void> {
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE)
    const jsonStr: string = (await pref.get(this.PREF_KEY_RULES, '')) as string

    if (jsonStr.trim().length === 0) {
      // 没存过：用默认 @State 的 notifyRules 即可，同时校准 nextRuleId
      this.recomputeNextRuleId()
      return
    }

    const parsed: NotifyRule[] = this.safeParseNotifyRules(jsonStr)
    if (parsed.length > 0) {
      this.notifyRules = parsed
      this.recomputeNextRuleId()
    }
  }

  private async saveNotifyRulesToPrefs(): Promise<void> {
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE)
    const jsonStr: string = JSON.stringify(this.notifyRules)
    await pref.put(this.PREF_KEY_RULES, jsonStr)
    await pref.flush()
  }

  private safeParseNotifyRules(jsonStr: string): NotifyRule[] {
    try {
      const raw: Object = JSON.parse(jsonStr) as Object
      if (!Array.isArray(raw)) return []

      const arr: Object[] = raw as Object[]
      const out: NotifyRule[] = []

      for (let i: number = 0; i < arr.length; i++) {
        const item: Record<string, Object> = arr[i] as Record<string, Object>

        const id: number = Number(item['id'])
        const stockCode: string = String(item['stockCode'] ?? '').trim()
        const stockName: string = String(item['stockName'] ?? '').trim()
        const conditionText: string = String(item['conditionText'] ?? '').trim()
        const enabled: boolean = Boolean(item['enabled'])

        // ✅ 新增：quoteType（老数据没有就默认 stock）
        const qtRaw: string = String(item['quoteType'] ?? 'stock').trim()
        const quoteType: QuoteType = (qtRaw === 'index') ? 'index' : 'stock'

        if (!Number.isNaN(id) && stockCode.length === 6 && stockName.length > 0 && conditionText.length > 0) {
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
      return []
    }
  }

  private recomputeNextRuleId(): void {
    let maxId: number = 0
    for (let i: number = 0; i < this.notifyRules.length; i++) {
      if (this.notifyRules[i].id > maxId) maxId = this.notifyRules[i].id
    }
    this.nextRuleId = maxId + 1
  }

  private async addNotifyRuleFromUI(): Promise<void> {
    const code: string = this.ruleStockCodeInput.trim()
    if (code.length !== 6) {
      this.ruleMsg = '请输入 6 位代码'
      return
    }

    const v: number = Number(this.rulePriceInput.trim())
    if (Number.isNaN(v)) {
      this.ruleMsg = '请输入正确的价格阈值'
      return
    }

    // 去重：同 code + 同条件 直接不加
    const opText: string = (this.ruleOpIndex === 0) ? '≥' : '≤'
    const cond: string = `价格 ${opText} ${v.toFixed(2)}`
    for (let i: number = 0; i < this.notifyRules.length; i++) {
      const r: NotifyRule = this.notifyRules[i]
      if (r.quoteType === this.ruleQuoteType && r.stockCode === code && r.conditionText === cond) {
        this.ruleMsg = '该规则已存在'
        return
      }
    }

    // 尽量从缓存/后端拿 name（你的 ensureQuoteCachedForCode 已有）
    const qt: QuoteType = this.ruleQuoteType
    await this.ensureQuoteCachedByType(qt, code)
    const q: Quote | null = this.findQuoteInCacheByType(qt, code)
    const name: string = (q !== null) ? q.name : code

    const newRule: NotifyRule = {
      id: this.nextRuleId,
      quoteType: qt,                 // ✅ 新增
      stockCode: code,
      stockName: name,
      conditionText: cond,
      enabled: true
    }
    this.nextRuleId += 1

    // ✅ 触发 UI：必须 new array
    const next: NotifyRule[] = [newRule].concat(this.notifyRules)
    this.notifyRules = next
    this.notifyRulesVer += 1
    await this.saveNotifyRulesToPrefs()

    // 清空输入
    this.ruleStockCodeInput = ''
    this.rulePriceInput = ''
    this.ruleMsg = '已添加并启用'
  }

  private async deleteRule(ruleId: number): Promise<void> {
    const next: NotifyRule[] = []
    for (let i: number = 0; i < this.notifyRules.length; i++) {
      const r: NotifyRule = this.notifyRules[i]
      if (r.id !== ruleId) next.push(r)
    }
    this.notifyRules = next
    this.notifyRulesVer += 1
    await this.saveNotifyRulesToPrefs()
  }

  private consumeFormIdFromAppStorageV2(from: string): void {
    // 从 AppStorageV2 里把桌面卡片传进来的 formId 取出来（只取一次，用完就清掉）
    const st: ActiveFormState | undefined =
      AppStorageV2.connect(ActiveFormState, APP_STORAGE_KEY_ACTIVE_FORM, () => new ActiveFormState())

    // AppStorage 取不到就直接返回（一般是 key 没注册成功或上下文异常）
    if (!st) {
      hilog.error(DOMAIN, TAG, '[%{public}s] AppStorageV2.connect failed', from)
      return
    }

    // formId 为空表示这次不是从卡片进来的，或者已经被消费过了
    const fid: string = st.formId
    if (fid.length === 0) {
      hilog.info(DOMAIN, TAG, '[%{public}s] no formId in ActiveFormState', from)
      return
    }

    // 保存到页面状态：后续更新桌面卡片时会用到这个 id
    this.activeFormId = fid

    // 清掉 AppStorage 里的值，避免 onPageShow/onResume 多次触发导致重复消费
    st.formId = ''

    hilog.info(DOMAIN, TAG, '[%{public}s] ✅ consumed formId=%{public}s -> this.activeFormId', from, fid)
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

  build() {

    Column() {
      // 顶部标题栏
      this.buildHeader()

      // ===== Tabs 部分 =====
      Tabs({ index: this.currentTabIndex }) {
        TabContent() {
          this.buildMarketTab()
        }
        .tabBar(this.tabBarBuilder('市场', 0))

        TabContent() {
          this.buildWatchlistTab()
        }
        .tabBar(this.tabBarBuilder('自选', 1))

        TabContent() {
          this.buildNotificationTab()
        }
        .tabBar(this.tabBarBuilder('通知', 2))
      }
      .barMode(BarMode.Fixed)
      .backgroundColor(Color.Black)
      .divider({
        strokeWidth: 0.5,
        color: '#333333'
      })
      .onChange((index: number) => {
        this.currentTabIndex = index;

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
    .height(50)                         // 区域更高，看起来更像大按钮区域
  }

  @Builder
  private buildModeChip(label: string, mode: 'index' | 'stock') {
    Text(label)
      .fontSize(16)
      .fontColor(this.currentMode === mode ? '#FFFFFF' : '#AAAAAA')
      .padding({ left: 10, right: 10, top: 4, bottom: 4 })
      .backgroundColor(this.currentMode === mode ? '#333333' : '#181818')
      .borderRadius(12)
      .onClick(() => {
        if (this.currentMode === mode) {
          return;
        }
        this.currentMode = mode;

        // 切模式时重置状态 + 重新拉数据
        this.marketQuotes = [];
        this.searchedQuotes = [];
        this.selectedQuoteCode = '';
        this.selectedQuoteName = '';
        this.selectedQuoteType = mode;
        this.selectedKlineIndex = -1;
        this.klinePoints = [];
        this.klineGeoms = [];
        this.klineXLabels = [];
        this.klineError = '';

        this.fetchQuotes(mode);
      })
  }


  // ====== Tab 1：市场（指数 / 股票 + K 线 + 桌面卡片预览） ======
  @Builder
  private buildMarketTab() {
    Column() {
      this.buildModeSwitcher()

      // 顶部：标题 + 模式切换 + 搜索区域
      Row() {
        Text(this.currentMode === 'index' ? '指数卡片' : '股票卡片')
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Color.White)
          .margin({ left: 16, top: 12, bottom: 4 })

        Blank()

        this.buildSearchArea();
      }
      .width('100%')
      .alignItems(VerticalAlign.Center)



      // 行情横向滚动 strip
      this.buildQuoteStrip();

      // K 线卡片
      this.buildKlineCard();



    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    .alignItems(HorizontalAlign.Start)
  }

  // ===== 行情 strip，带加载 / 错误状态 + 可删除卡片 =====
  @Builder
  private buildQuoteStrip() {
    if (this.listLoading) {
      Row() {
        Text('行情加载中…')
          .fontSize(12)
          .fontColor('#888888')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    } else if (this.listError) {
      Row() {
        Text(`行情加载失败：${this.listError}`)
          .fontSize(12)
          .fontColor('#FF7875')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    } else {
      Scroll() {
        Row({ space: 12 }) {
          // 搜索出来的卡片
          ForEach(this.searchedQuotes, (item: Quote) => {
            this.buildQuoteCard(item);
          }, (item: Quote) => `search-${item.type}-${item.code}`)

          // 后台实时卡片
          ForEach(this.marketQuotes, (item: Quote) => {
            this.buildQuoteCard(item);
          }, (item: Quote) => `rt-${item.type}-${item.code}`)
        }
        .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      }
      .scrollable(ScrollDirection.Horizontal)
    }
  }

  // ===== 搜索区域 =====
  @Builder
  private buildSearchArea() {
    Column() {
      Row({ space: 8 }) {
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
          .onChange((value: string) => {
            this.searchCode = value.trim();
          })

        Button('查询')
          .height(32)
          .onClick(() => {
            this.queryQuoteByCode();
          })
      }
      .padding({ left: 16, right: 16, top: 8, bottom: 4 })

      if (this.searchLoading) {
        Row() {
          Text('正在查询…')
            .fontSize(12)
            .fontColor('#888888')
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 8 })
      } else if (this.searchError) {
        Row() {
          Text(`查询失败：${this.searchError}`)
            .fontSize(12)
            .fontColor('#FF7875')
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 8 })
      }
    }
  }

  private toggleCardExpanded(item: Quote): void {
    const key: string = `${item.type}-${item.code}`;
    const oldVal: boolean = this.cardExpandedMap.get(key) ?? false;

    // 复制新 Map（触发 State 更新）
    const newMap: Map<string, boolean> = new Map<string, boolean>();
    this.cardExpandedMap.forEach((value: boolean, k: string) => {
      newMap.set(k, value);
    });

    newMap.set(key, !oldVal);
    this.cardExpandedMap = newMap;
  }




  // ===== 行情卡片（指数 / 股票通用） =====
  @Builder
  private buildQuoteCard(item: Quote) {
    Column() {
      Column() {
        // ===== 卡片主体：点这里才选中 + 拉 K 线 =====
        Column() {
          Row() {
            Text(item.name)
              .fontSize(12)
              .fontColor('#AAAAAA')

            Text('✕')
              .fontSize(12)
              .fontColor('#777777')
              // ✅ 用 TapGesture，避免被 Scroll 手势吞 / 冲突
              .parallelGesture(
                TapGesture().onAction(async () => {
                  await this.removeQuoteCard(item.code, item.type)
                })
              )
          }
          .width('100%')
          .justifyContent(FlexAlign.SpaceBetween)
          .alignItems(VerticalAlign.Center)

          Text(item.price.toFixed(2))
            .fontSize((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 28 : 20)
            .fontWeight(FontWeight.Bold)
            .fontColor(item.change >= 0 ? '#FF4D4F' : '#39D98A')
            .margin({ top: 4 })

          Text(`${item.change.toFixed(2)}  ${item.changePercent.toFixed(2)}%`)
            .fontSize((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 16 : 12)
            .fontColor(item.change >= 0 ? '#FF7875' : '#6BE49B')
            .margin({ top: 4 })
        }

        // ===== 右下角 +/-：只负责展开收起 =====
        Row() {
          Blank()
          Text((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? '－' : '＋')
            .fontSize(18)
            .fontColor('#CCCCCC')
            // ✅ 也用 TapGesture，避免和外层点击打架
            .parallelGesture(
              TapGesture().onAction(() => {
                this.toggleCardExpanded(item)
              })
            )
        }
        .width('100%')
        .margin({ top: 8 })
      }
      .padding((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 20 : 12)
      .backgroundColor('#171717')
      .borderRadius(12)
      .width((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 200 : 140)
      .height((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 150 : undefined)

      // ✅ 关键：外层卡片用 parallelGesture 捕获 tap，横向 Scroll 下更稳定
      .parallelGesture(
        TapGesture().onAction(async () => {
          console.info(`[TAP] quoteCard tapped code=${item.code} activeFormId=${this.activeFormId}`)
          hilog.error(DOMAIN, TAG, '[TAP] quoteCard tapped code=%{public}s activeFormId=%{public}s',
            item.code, this.activeFormId)

          this.selectedQuoteCode = item.code
          this.selectedQuoteName = item.name
          this.selectedQuoteType = item.type
          hilog.error(DOMAIN, TAG, '[TAP] selected set type=%{public}s code=%{public}s', item.type, item.code)

          hilog.error(DOMAIN, TAG, '[TAP] before updateWidgetIfLaunchedFromForm')
          await this.updateWidgetIfLaunchedFromForm(item)
          hilog.error(DOMAIN, TAG, '[TAP] after updateWidgetIfLaunchedFromForm')

          // ✅ 你要的“点击切换行情卡片 + 拉K线”
          const period: string = this.getKlinePeriodParam()
          this.fetchKline(item.type, item.code, period)
        })
      )
    }
  }


  // ===== K 线卡片 =====
  @Builder
  private buildKlineCard() {
    Column() {
      // 标题 + 周期切换
      Row({ space: 12 }) {
        Text(
          this.selectedQuoteName !== ''
            ? `${this.selectedQuoteName} K 线`
            : (this.currentMode === 'index' ? '指数 K 线' : '股票 K 线')
        )
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Color.White)

        Blank()

        this.buildKlinePeriodChip('日K', 0)
        this.buildKlinePeriodChip('周K', 1)
        this.buildKlinePeriodChip('月K', 2)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 8 })

      // 当前选中 K 线的详细信息条
      if (this.selectedKlineIndex >= 0) {
        Row({ space: 8 }) {
          Text(this.selectedKlineDate)
            .fontSize(10)
            .fontColor('#CCCCCC')

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

      // 主体区域
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
          Column() {
            Row() {
              // y 轴刻度
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

              // 右侧：K 线 + x 轴一起横向滚动
              Scroll() {
                Column() {
                  // 蜡烛图区域
                  Row({ space: this.candleGap }) {
                    ForEach(this.klineGeoms, (g: KlineGeom, idx: number) => {
                      Column() {
                        Blank().height(g.topGap)

                        Rect()      // 上影线
                          .width(2)
                          .height(g.upperWick)
                          .fill('#444444')

                        Rect()      // 实体
                          .width(this.candleBodyWidth)
                          .height(g.bodyHeight)
                          .fill(g.isUp ? '#39D98A' : '#FF4D4F')

                        Rect()      // 下影线
                          .width(2)
                          .height(g.lowerWick)
                          .fill('#444444')

                        Blank().height(4)
                      }
                      .border({
                        width: this.selectedKlineIndex === idx ? 1 : 0,
                        color: this.selectedKlineIndex === idx ? '#FACC15' : '#000000'
                      })
                      .parallelGesture(
                        TapGesture().onAction(() => {
                          this.onKlineSelected(idx)
                        }))
                    }, (g: KlineGeom, idx: number) => `${idx}`)
                  }
                  .width(this.getKlineTotalWidth())
                  .height(this.KLINE_PLOT_H)

                  // x 轴日期标签
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
                .padding({ right: 32 })
              }
              .scrollable(ScrollDirection.Horizontal)
              .height(this.KLINE_PLOT_H + this.KLINE_LABEL_H)
              .margin({ left: 8 })
            }
            .width('100%')
          }
        }
      }
      .width('100%')
      .height(this.KLINE_PLOT_H + this.KLINE_LABEL_H + 70)
      .margin({ top: 8, bottom: 12 })
      .padding(16)
      .backgroundColor('#101010')
      .borderRadius(16)
    }
  }

  @Builder
  private buildKlinePeriodChip(label: string, index: number) {
    Text(label)
      .fontSize(12)
      .fontColor(this.klinePeriodIndex === index ? '#FFFFFF' : '#AAAAAA')
      .padding({ left: 10, right: 10, top: 4, bottom: 4 })
      .backgroundColor(this.klinePeriodIndex === index ? '#333333' : '#181818')
      .borderRadius(12)
      .onClick(() => {
        if (this.klinePeriodIndex === index) {
          return;
        }
        this.klinePeriodIndex = index;

        if (this.selectedQuoteCode !== '') {
          const period: string = this.getKlinePeriodParam();
          this.fetchKline(this.selectedQuoteType, this.selectedQuoteCode, period);
        }
      })
  }


  @Builder
  private buildDesktopCardSmall() {
    Column() {
      Text('自选股（小号卡片）')
        .fontSize(10)
        .fontColor('#999999')
        .margin({ bottom: 4 })

      Text('300750 宁德时代')
        .fontSize(12)
        .fontColor(Color.White)

      Text('210.35  -1.25  -0.59%')
        .fontSize(11)
        .fontColor('#39D98A')
        .margin({ top: 4 })
    }
    .padding(10)
    .backgroundColor('#151515')
    .borderRadius(14)
    .width(140)
  }

  @Builder
  private buildDesktopCardLarge() {
    Column() {
      Row() {
        Text('大盘指数')
          .fontSize(12)
          .fontColor('#BBBBBB')
        Blank()
        Text('上证')
          .fontSize(11)
          .fontColor('#888888')
      }

      Text('3050.32')
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor('#FF4D4F')
        .margin({ top: 4 })

      Text('+12.34  +0.41%')
        .fontSize(11)
        .fontColor('#FF7875')
        .margin({ top: 2 })

      Row({ space: 2 }) {
        ForEach([16, 10, 18, 8, 20, 12, 14], (h: number, idx: number) => {
          Rect()
            .width(6)
            .height(h)
            .fill('#333333')
        }, (h: number, idx: number) => `${idx}-${h}`)
      }
      .margin({ top: 8 })
    }
    .padding(12)
    .backgroundColor('#151515')
    .borderRadius(14)
    .width(180)
  }

  // ====== Tab 2：自选（分组管理 + 组内列表） ======
  @Builder
  private buildWatchlistTab() {
    Column() {
      this.buildGroupStrip()
      this.buildCreateGroupBar()

      if (this.watchlistMsg.length > 0) {
        Text(this.watchlistMsg)
          .fontSize(10)
          .fontColor('#777777')
          .margin({ left: 16, bottom: 8 })
      }

      if (this.activeGroupId < 0 || this.groups.length === 0) {
        Text('暂无分组，请先创建一个分组')
          .fontSize(12)
          .fontColor('#777777')
          .margin({ top: 24 })
          .width('100%')
          .textAlign(TextAlign.Center)
      } else {
        // 组头（无参 Builder：直接用 activeGroupName + activeGroupId）
        this.buildActiveGroupHeaderCurrent()

        // 统计（无参 Builder：用 activeGroupXXXText）
        this.buildGroupStatsCard()

        // 添加条
        this.buildAddStockBar()

        if (this.getActiveGroupStocksList().length === 0) {
          Text('该分组暂无股票，请在下方输入代码添加')
            .fontSize(12)
            .fontColor('#666666')
            .margin({ top: 12, left: 16 })
        } else {
          List() {
            ForEach(this.activeStocks, (s: StockRef) => {
              this.buildStockRefRow(this.activeGroupId, s)
            }, (s: StockRef) => {
              // key 只需要稳定唯一即可，不要把 ver 放这里也行
              return `${this.activeGroupId}-${s.code}`
            })
          }
          .key(`${this.activeGroupId}-${this.activeStocksVer}`)  // ✅关键：让 List 重新创建
          .padding({ left: 16, right: 16, top: 8, bottom: 12 })
        }
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    .key(`watchlist-${this.activeGroupId}-${this.activeStocksVer}-${this.quoteCacheVer}-${this.activeStatsVer}`)
  }

  @Builder
  private buildActiveGroupHeaderCurrent() {
    Row({ space: 8 }) {
      Text(this.getActiveGroupNameText())
        .fontSize(14)
        .fontColor('#FFFFFF')
        .fontWeight(FontWeight.Medium)

      Blank()

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


  @Builder
  private buildGroupStrip() {
    Scroll() {
      Row({ space: 8 }) {
        ForEach(this.groups, (group: StockGroup) => {
          Column({ space: 2 }) {
            Text(group.name)
              .fontSize(12)
              .fontColor(this.activeGroupId === group.id ? '#FFFFFF' : '#DDDDDD')

            Text(this.getGroupAvgPctText(group.id))
              .fontSize(10)
              .fontColor(this.getGroupAvgPctColor(group.id))
          }
          .padding({ left: 12, right: 12, top: 6, bottom: 6 })
          .backgroundColor(this.activeGroupId === group.id ? '#2A2A2A' : '#161616')
          .borderRadius(12)
          .onClick(async () => {
            this.activeGroupId = group.id

            this.syncActiveStocks()
            await this.preloadActiveGroupQuotes()
            this.syncActiveStocks()
          })
        }, (group: StockGroup) => group.id.toString())
      }
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })
    }
    .scrollable(ScrollDirection.Horizontal)
  }

  @Builder
  private buildCreateGroupBar() {
    Row({ space: 8 }) {
      TextInput({
        placeholder: '新建分组名（如 科技股）',
        text: this.newGroupName // ✅ 受控：UI 绑定 state
      })
        .width('68%')
        .height(32)
        .fontSize(13)
        .backgroundColor(Color.White)
        .fontColor(Color.Black)
        .onChange((v: string) => {
          this.newGroupName = v.trim();
        })

      Button('创建')
        .height(32)
        .onClick(async () => {
          const name: string = this.newGroupName.trim();
          if (name.length === 0) {
            this.watchlistMsg = '请输入分组名';
            return;
          }
          await this.createGroup(name);

        })
    }
    .padding({ left: 16, right: 16, bottom: 8 })
  }


  @Builder
  private buildActiveGroupHeader(g: StockGroup) {
    Row({ space: 8 }) {
      Text(g.name)
        .fontSize(14)
        .fontColor('#FFFFFF')
        .fontWeight(FontWeight.Medium)

      Blank()

      Text('删除分组')
        .fontSize(12)
        .fontColor('#FF7875')
        .padding({ left: 10, right: 10, top: 6, bottom: 6 })
        .backgroundColor('#1A1A1A')
        .borderRadius(10)
        .onClick(() => {
          this.deleteGroup(g.id)
        })
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 8, bottom: 6 })
  }



  @Builder
  private buildGroupStatsCard() {
    Column() {
      Row({ space: 12 }) {

        Column({ space: 4 }) {
          Text('股票数').fontSize(10).fontColor('#777777')
          Text(this.activeStatStockCount).fontSize(14).fontColor('#FFFFFF').fontWeight(FontWeight.Medium)
        }.width('22%')

        Column({ space: 4 }) {
          Text('上涨').fontSize(10).fontColor('#777777')
          Text(this.activeStatUp).fontSize(14).fontColor('#FFFFFF').fontWeight(FontWeight.Medium)
        }.width('22%')

        Column({ space: 4 }) {
          Text('下跌').fontSize(10).fontColor('#777777')
          Text(this.activeStatDown).fontSize(14).fontColor('#FFFFFF').fontWeight(FontWeight.Medium)
        }.width('22%')

        Column({ space: 4 }) {
          Text('均涨跌').fontSize(10).fontColor('#777777')
          Text(this.activeStatAvgPct).fontSize(14).fontColor('#FFFFFF').fontWeight(FontWeight.Medium)
        }.width('22%')
      }
      .width('100%')
      .padding(12)
      .margin({ left: 16, right: 16, bottom: 8 })
      .backgroundColor('#101010')
      .borderRadius(14)
    }
    // 这个 key 你保留即可
    .key(`stats-${this.activeGroupId}-${this.activeStocksVer}-${this.quoteCacheVer}-${this.activeStatsVer}`)
  }


  @Builder
  private buildAddStockBar() {
    Row({ space: 8 }) {
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
          this.addStockCodeInput = v.trim();
        })

      Button('添加')
        .height(32)
        .onClick(async () => {
          const code: string = this.addStockCodeInput.trim();
          if (code.length !== 6) {
            this.watchlistMsg = '请输入 6 位代码';
            return;
          }

          // ✅ 关键：永远取“当前 activeGroupId”，不再依赖 Builder 参数闭包
          const gid: number = this.activeGroupId;
          if (gid < 0) {
            this.watchlistMsg = '当前没有可用分组';
            return;
          }

          await this.addStocksToGroup(gid, [code]);
          this.watchlistMsg = `已尝试添加：${code} -> group=${gid}`;
        })
    }
    .padding({ left: 16, right: 16, bottom: 8 })
  }



  @Builder
  private buildStockRefRow(groupId: number, s: StockRef) {
    ListItem() {
      Row() {
        Column({ space: 4 }) {
          Text(this.getStockRowName(s))
            .fontSize(14)
            .fontColor(Color.White)

          Text(s.code)
            .fontSize(11)
            .fontColor('#666666')
        }

        Blank()

        Column({ space: 4 }) {
          Text(this.getStockRowPriceText(s))
            .fontSize(14)
            .fontColor(this.getStockRowPriceColor(s))
            .textAlign(TextAlign.End)

          Text(this.getStockRowChangeText(s) + '  ' + this.getStockRowPctText(s))
            .fontSize(11)
            .fontColor(this.getStockRowSubColor(s))
            .textAlign(TextAlign.End)
        }

        Button('✕')
          .height(28)
          .width(28)
          .padding(0)
          .backgroundColor('#1A1A1A')
          .fontColor('#777777')
          .borderRadius(14)
          .onClick(() => {
            // 只做删除，不要 async，避免手势链路更复杂
            this.removeStocksFromGroup(groupId, [s.code])
          })
      }
      .padding({ left: 12, right: 12, top: 10, bottom: 10 })
      .backgroundColor('#101010')
      .borderRadius(12)
    }
    .margin({ bottom: 8 })
  }


  @Builder
  private buildWatchlistRow(s: StockQuote) {
    ListItem() {
      Row() {
        Column({ space: 4 }) {
          Text(s.name)
            .fontSize(14)
            .fontColor(Color.White)

          Text(s.code)
            .fontSize(11)
            .fontColor('#666666')
        }

        Blank()

        Column({ space: 4 }) {
          Text(s.price.toFixed(2))
            .fontSize(14)
            .fontColor(s.change >= 0 ? '#FF4D4F' : '#39D98A')
            .textAlign(TextAlign.End)

          Text(`${s.change.toFixed(2)}  ${s.changePercent.toFixed(2)}%`)
            .fontSize(11)
            .fontColor(s.change >= 0 ? '#FF7875' : '#6BE49B')
            .textAlign(TextAlign.End)
        }
      }
      .padding({ left: 16, right: 16, top: 10, bottom: 10 })
      .backgroundColor('#101010')
    }
    .margin({ bottom: 1 })
  }

  // ====== Tab 3：行情通知中心（只是 UI 列表） ======
  @Builder
  private buildNotificationTab() {
    Column() {
      Row() {
        Text('通知规则').fontSize(14).fontColor(Color.White)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })

      // ✅ 新增规则区
      Column({ space: 8 }) {
        Column({space:8}){
          Row({ space: 6 }) {
            this.buildRuleTypeChip('股票', 'stock')
            this.buildRuleTypeChip('指数', 'index')
          }
          Row({ space: 8 }) {
            // ✅ 类型选择：股票 / 指数


            // ✅ 代码输入
            TextInput({
              placeholder: this.ruleQuoteType === 'stock' ? '股票代码(6位)' : '指数代码(6位)',
              text: this.ruleStockCodeInput
            })
              .width('28%')
              .height(32)
              .fontSize(13)
              .backgroundColor(Color.White)
              .fontColor(Color.Black)
              .onChange((v: string) => { this.ruleStockCodeInput = v.trim() })

            // ✅ 操作符
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

            // ✅ 阈值
            TextInput({ placeholder: '阈值', text: this.rulePriceInput })
              .width('18%')
              .height(32)
              .fontSize(13)
              .backgroundColor(Color.White)
              .fontColor(Color.Black)
              .onChange((v: string) => { this.rulePriceInput = v.trim() })

            // ✅ 添加
            Button('添加')
              .height(32)
              .onClick(async () => { await this.addNotifyRuleFromUI() })
          }
        }


        if (this.ruleMsg.length > 0) {
          Text(this.ruleMsg)
            .fontSize(10)
            .fontColor('#777777')
        }
      }
      .padding({ left: 16, right: 16, bottom: 10 })

      // ✅ 规则列表
      List() {
        ForEach(this.notifyRules, (rule: NotifyRule) => {
          this.buildNotifyRuleItem(rule)
        }, (rule: NotifyRule) => rule.id.toString())
      }
      .key(`notifyRules-${this.notifyRulesVer}`)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }

  @Builder
  private buildNotifyRuleItem(rule: NotifyRule) {
    ListItem() {
      Row({ space: 10 }) {
        Column({ space: 4 }) {
          Row({ space: 6 }) {
            // ✅ 类型标识
            Text(rule.quoteType === 'stock' ? '股票' : '指数')
              .fontSize(10)
              .fontColor('#AAAAAA')
              .padding({ left: 6, right: 6, top: 2, bottom: 2 })
              .backgroundColor('#1A1A1A')
              .borderRadius(6)

            Text(`${rule.stockName}（${rule.stockCode}）`)
              .fontSize(13)
              .fontColor(Color.White)
          }

          Text(rule.conditionText)
            .fontSize(11)
            .fontColor('#BBBBBB')
        }
        .layoutWeight(1)

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
    .margin({ bottom: 1 })
  }

  @Builder
  private buildRuleTypeChip(label: string, t: 'index' | 'stock') {
    Text(label)
      .fontSize(12)
      .fontColor(this.ruleQuoteType === t ? '#FFFFFF' : '#AAAAAA')
      .padding({ left: 10, right: 10, top: 6, bottom: 6 })
      .backgroundColor(this.ruleQuoteType === t ? '#333333' : '#181818')
      .borderRadius(10)
      .onClick(() => { this.ruleQuoteType = t })
  }
}
