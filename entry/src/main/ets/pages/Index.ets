// entry/src/main/ets/pages/Index.ets
import { TabBarOptions } from '@kit.ArkUI';
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




// ====== 通用行情结构（指数 / 股票） ======
interface Quote {
  code: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  type: 'index' | 'stock';
}

interface BackendQuote {
  code: string;
  name: string;
  price: number;
  change: number;
  change_percent: number;
}

// ====== 自选股 / 通知用结构（保持你原来的） ======
interface StockQuote {
  name: string;
  code: string;
  price: number;
  change: number;
  changePercent: number;
}

interface NotifyRule {
  id: number;
  stockCode: string;
  stockName: string;
  conditionText: string; // 比如 "价格 ≥ 25.00"
  enabled: boolean;
}

// ====== K 线点结构（指数 / 股票通用） ======
interface KLinePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface KlineGeom {
  upperWick: number;   // 上影线高度
  bodyHeight: number;  // 实体高度
  lowerWick: number;   // 下影线高度
  topGap: number;      // 整根蜡烛距离顶部的空白
  isUp: boolean;       // 是否上涨（close >= open）
}


interface StockRef {
  code: string;      // 6 位
  name?: string;     // 可选：第一次加入时填；后续可由后端 quote 补全
}

interface StockGroup {
  id: number;
  name: string;
  stocks: StockRef[];
  createdAt: number;
  version: number; // ✅ 用于强制 List 识别结构变化
}

interface GroupStats {
  groupId: number;
  stockCount: number;
  upCount: number;        // 当日上涨数量（需要行情数据）
  downCount: number;      // 当日下跌数量
  avgChangePercent: number; // 平均涨跌幅（需要行情数据）
}

interface ActiveGroupStats {
  stockCount: string;
  up: string;
  down: string;
  avgPct: string;
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


// ====== 页面组件 ======
@Entry
@Component
struct HarmonyStockPage {
  // 顶部 Tab：0 市场 1 自选 2 通知
  @State currentTabIndex: number = 0;

  // 当前行情模式：指数 / 股票
  @State currentMode: 'index' | 'stock' = 'index';

  // 行情列表状态（strip）
  @State listLoading: boolean = false;
  @State listError: string = '';

  // 搜索状态
  @State searchCode: string = '';
  @State searchLoading: boolean = false;
  @State searchError: string = '';

  // 行情卡片：后台实时 + 搜索结果
  @State marketQuotes: Quote[] = [];
  @State searchedQuotes: Quote[] = [];

  // 当前选中的标的（用于 K 线）
  @State selectedQuoteCode: string = '';
  @State selectedQuoteName: string = '';
  @State selectedQuoteType: 'index' | 'stock' = 'index';

  // 当前 K 线周期：0 日K 1 周K 2 月K
  @State klinePeriodIndex: number = 0;

  // K 线请求状态
  @State klineLoading: boolean = false;
  @State klineError: string = '';
  @State klinePoints: KLinePoint[] = [];

  // K 线几何信息 & 坐标轴
  @State klineGeoms: KlineGeom[] = [];
  @State klineMinPrice: number = 0;
  @State klineMaxPrice: number = 0;
  @State klineXLabels: string[] = [];

  private readonly KLINE_PLOT_H: number = 180;     // 蜡烛绘图区高度（计算 + 渲染统一）
  private readonly KLINE_LABEL_H: number = 24;     // x轴标签区域高度
  private readonly KLINE_CARD_INNER_PAD: number = 16;


  // 当前选中的 K 线（用于显示详细信息条 + 高亮）
  @State selectedKlineIndex: number = -1;
  @State selectedKlineDate: string = '';
  @State selectedKlineOpen: number = 0;
  @State selectedKlineHigh: number = 0;
  @State selectedKlineLow: number = 0;
  @State selectedKlineClose: number = 0;

  // 每根 K 线实体宽度 & 间距
  private candleBodyWidth: number = 8;
  private candleGap: number = 4;

  @State expandedKeys: string[] = [];
  // 记录每张卡片是否展开：key = `${type}-${code}`
  @State cardExpandedMap: Map<string, boolean> = new Map<string, boolean>();

  private readonly HTTP_CONNECT_TIMEOUT: number = 55000; // 55s
  private readonly HTTP_READ_TIMEOUT: number = 55000;    // 55s

  private readonly PREF_FILE: string = 'harmonystock_watchlist';
  private readonly PREF_KEY_GROUPS: string = 'groups_json';
  private nextGroupId: number = 1;

  @State activeGroupId: number = -1; // 当前选中的分组
  @State watchlistMsg: string = '';
  @State newGroupName: string = '';
  @State addStockCodeInput: string = '';

  @State quoteCache: Map<string, Quote> = new Map<string, Quote>();
  @State quoteCacheVer: number = 0

  @State activeStocks: StockRef[] = []
  @State activeStocksVer: number = 0

  @State activeStatStockCount: string = '0'
  @State activeStatUp: string = '0'
  @State activeStatDown: string = '0'
  @State activeStatAvgPct: string = '--'
  @State activeStatsVer: number = 0   // 保留也行（用于 key）

  // ✅ 加在这里（和其他状态字段放一起）
  @State activeFormId: string = ''

  @State ruleStockCodeInput: string = ''
  @State ruleOpIndex: number = 0        // 0: ≥  1: ≤
  @State rulePriceInput: string = ''    // 阈值
  @State ruleMsg: string = ''

  @State notifyRulesVer: number = 0

  onPageShow() {
    hilog.error(0x0000, 'INDEX_DEBUG', '🔥 Index onPageShow CALLED')

    // ① 先消费 formId（你已有）
    this.consumeFormIdFromAppStorageV2('onPageShow')

    // ② 如果已经有选中的行情，主动同步到桌面卡片
    if (this.activeFormId.length > 0 && this.selectedQuoteCode.length > 0) {
      const q: Quote | null = this.findQuoteInCaches(this.selectedQuoteCode)
      if (q !== null) {
        hilog.error(DOMAIN, TAG,
          '[WIDGET] onPageShow sync current quote code=%{public}s',
          q.code
        )
        this.updateWidgetIfLaunchedFromForm(q)
      }
    }
  }

  private consumeFormIdFromAppStorageV2(from: string): void {
    const st: ActiveFormState | undefined =
      AppStorageV2.connect(ActiveFormState, APP_STORAGE_KEY_ACTIVE_FORM, () => new ActiveFormState())

    if (!st) {
      hilog.error(DOMAIN, TAG, '[%{public}s] AppStorageV2.connect failed', from)
      return
    }

    const fid: string = st.formId
    if (fid.length === 0) {
      hilog.info(DOMAIN, TAG, '[%{public}s] no formId in ActiveFormState', from)
      return
    }

    // 1) 同步到页面字段：后续 updateWidgetIfLaunchedFromForm 依赖它
    this.activeFormId = fid

    // 2) 消费掉，避免重复
    st.formId = ''

    hilog.info(DOMAIN, TAG, '[%{public}s] ✅ consumed formId=%{public}s -> this.activeFormId', from, fid)
  }


  private refreshActiveGroupStats(): void {
    const st: ActiveGroupStats = this.computeStatsFromStocks(this.activeStocks)
    this.activeStatStockCount = st.stockCount
    this.activeStatUp = st.up
    this.activeStatDown = st.down
    this.activeStatAvgPct = st.avgPct
    this.activeStatsVer += 1
    console.info(`[DBG] stats=${this.activeStatStockCount}, up=${this.activeStatUp}, down=${this.activeStatDown}, avg=${this.activeStatAvgPct}, ver=${this.activeStatsVer}`)

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

  @State notifyRules: NotifyRule[] = [
    { id: 999, stockCode: '300750', stockName: '宁德时代', conditionText: '价格 ≥ 1.00', enabled: true },
    { id: 2, stockCode: '601012', stockName: '隆基绿能', conditionText: '价格 ≤ 28.00', enabled: false }
  ]

  private nextRuleId: number = 1000
  private readonly PREF_KEY_RULES: string = 'notify_rules_json'

  // ===== 通用：后端行情 -> 前端 Quote =====
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

  // ===== 页面生命周期 =====
  aboutToAppear() {
    this.dbg('aboutToAppear:enter')

    this.loadGroupsFromPrefs().then(async () => {
      await this.loadNotifyRulesFromPrefs()   // ✅ 加这一句
      await this.preloadActiveGroupQuotes()
      this.fetchQuotes(this.currentMode)
    })

    RuleEngine.start((): NotifyRuleLite[] => {
      const out: NotifyRuleLite[] = []
      for (let i: number = 0; i < this.notifyRules.length; i++) {
        const r: NotifyRule = this.notifyRules[i]
        out.push({ id: r.id, stockCode: r.stockCode, stockName: r.stockName, conditionText: r.conditionText, enabled: r.enabled })
      }
      return out
    }, 5000, 10000)
  }
  aboutToDisappear(): void {
    RuleEngine.stop()
  }

  // ===== 拉取行情列表（指数 / 股票共用） =====
  private async fetchQuotes(type: 'index' | 'stock'): Promise<void> {
    const affectUI: boolean = (type === this.currentMode)

    if (affectUI) {
      this.listLoading = true
      this.listError = ''
    }

    const httpRequest = http.createHttp()
    try {
      const url: string = `http://101.43.185.73:8000/api/quote?type=${type}`

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      if (res.responseCode === 200 && res.result) {
        const raw: BackendQuote[] = JSON.parse(res.result as string) as BackendQuote[]
        const mapped: Quote[] = raw.map((it: BackendQuote): Quote => this.mapBackendQuote(it, type))

        // 当前模式：更新市场列表
        if (affectUI) {
          this.marketQuotes = mapped
        }

        // 两种模式：都写入缓存（自选统计依赖）
        this.mergeQuotesToCache(mapped)

        // 默认选中第一只 + 拉K线：只对当前模式生效
        if (affectUI && this.selectedQuoteCode === '' && mapped.length > 0) {
          const first: Quote = mapped[0]
          this.selectedQuoteCode = first.code
          this.selectedQuoteName = first.name
          this.selectedQuoteType = first.type
          const period: string = this.getKlinePeriodParam()
          this.fetchKline(first.type, first.code, period)
        }
      } else {
        if (affectUI) {
          this.listError = `HTTP ${res.responseCode}`
          this.marketQuotes = []
        }
        // 非当前模式失败：不要污染 UI
      }
    } catch (e) {
      if (affectUI) {
        this.listError = `${e}`
        this.marketQuotes = []
      }
    } finally {
      httpRequest.destroy()
      if (affectUI) {
        this.listLoading = false
      }
    }
  }



  // ===== 拉取 K 线数据（指数 / 股票共用） =====
  private async fetchKline(
    type: 'index' | 'stock',
    code: string,
    period: string
  ): Promise<void> {
    this.klineLoading = true;
    this.klineError = '';

    const httpRequest = http.createHttp();
    try {
      const url: string =
        `http://101.43.185.73:8000/api/kline` +
          `?type=${type}&code=${code}&period=${period}&limit=60`;

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      });


      if (res.responseCode === 200 && res.result) {
        const raw: KLinePoint[] =
          JSON.parse(res.result as string) as KLinePoint[];

        this.klinePoints = raw;

        // 默认选中最后一根 K 线
        if (this.klinePoints.length > 0) {
          const lastIndex: number = this.klinePoints.length - 1;
          this.onKlineSelected(lastIndex);
        } else {
          this.selectedKlineIndex = -1;
        }

        this.recomputeKlineGeom();
      } else {
        this.klineError = `HTTP ${res.responseCode}`;
        this.klinePoints = [];
        this.klineGeoms = [];
        this.klineXLabels = [];
        this.selectedKlineIndex = -1;
      }
    } catch (e) {
      this.klineError = `${e}`;
      this.klinePoints = [];
      this.klineGeoms = [];
      this.klineXLabels = [];
      this.selectedKlineIndex = -1;
    } finally {
      httpRequest.destroy();
      this.klineLoading = false;
    }
  }

  // ===== 根据当前周期索引 -> period 参数 =====
  private getKlinePeriodParam(): string {
    if (this.klinePeriodIndex === 1) {
      return 'week';   // 周 K
    }
    if (this.klinePeriodIndex === 2) {
      return 'month';  // 月 K
    }
    return 'day';      // 默认日 K
  }

  // ===== 计算 K 线几何信息 & 坐标轴 =====
  private recomputeKlineGeom(): void {
    const points: KLinePoint[] = this.klinePoints;
    const geoms: KlineGeom[] = [];
    const labels: string[] = [];

    if (points.length === 0) {
      this.klineGeoms = geoms;
      this.klineXLabels = labels;
      return;
    }

    // 1) 找最高价 / 最低价
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

    // range 防御
    let range: number = maxPrice - minPrice;
    if (range <= 0) {
      range = 1.0;
      maxPrice = minPrice + range;
    }

    this.klineMinPrice = minPrice;
    this.klineMaxPrice = maxPrice;

    // 2) 绘图区高度统一使用常量（必须和 UI 的蜡烛区域 .height 一致）
    const fullHeight: number = this.KLINE_PLOT_H;

    // 最小可见高度（防止“十字线”看不到）
    const minBody: number = 4;
    const minWick: number = 2;

    // 价格 -> y 坐标（0 顶部，fullHeight 底部）
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

    // 3) 逐个点算几何
    for (let i: number = 0; i < points.length; i++) {
      const pt: KLinePoint = points[i];

      // 将关键价格映射到 y
      const yHigh: number = priceToY(pt.high);
      const yLow: number = priceToY(pt.low);
      const yOpen: number = priceToY(pt.open);
      const yClose: number = priceToY(pt.close);

      // 实体上下边界
      let yTopBody: number = Math.min(yOpen, yClose);
      let yBottomBody: number = Math.max(yOpen, yClose);

      // 实体最小高度保证
      if (yBottomBody - yTopBody < minBody) {
        const mid: number = (yTopBody + yBottomBody) / 2.0;
        yTopBody = mid - minBody / 2.0;
        yBottomBody = mid + minBody / 2.0;

        // 裁剪到绘图区范围
        if (yTopBody < 0) {
          yTopBody = 0;
          yBottomBody = minBody;
        }
        if (yBottomBody > fullHeight) {
          yBottomBody = fullHeight;
          yTopBody = fullHeight - minBody;
        }
      }

      // 上影线：high -> 实体上边
      let upperWick: number = yTopBody - yHigh;
      if (upperWick < minWick) {
        upperWick = minWick;
      }

      // 下影线：实体下边 -> low
      let lowerWick: number = yLow - yBottomBody;
      if (lowerWick < minWick) {
        lowerWick = minWick;
      }

      // 顶部空白 = yHigh
      let topGap: number = yHigh;
      if (topGap < 0) {
        topGap = 0;
      }
      if (topGap > fullHeight) {
        topGap = fullHeight;
      }

      // 额外防御：避免 topGap + wick + body 超出容器导致溢出
      // 如果确实溢出，优先压缩 topGap（不动实体/影线）
      const totalH: number = topGap + upperWick + (yBottomBody - yTopBody) + lowerWick;
      if (totalH > fullHeight) {
        const overflow: number = totalH - fullHeight;
        topGap = topGap - overflow;
        if (topGap < 0) {
          topGap = 0;
        }
      }

      geoms.push({
        upperWick: upperWick,
        bodyHeight: (yBottomBody - yTopBody),
        lowerWick: lowerWick,
        topGap: topGap,
        isUp: pt.close >= pt.open
      });
    }

    // 4) x 轴标签：最多 5 个，均匀取样
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

        const dateStr: string = points[idx].date;
        const label: string = dateStr.length >= 5 ? dateStr.substring(5) : dateStr;
        labels.push(label);
      }
    }

    this.klineGeoms = geoms;
    this.klineXLabels = labels;
  }


  // ===== 选中某一根 K 线 =====
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
    this.searchedQuotes = this.searchedQuotes.filter((item: Quote) =>
    !(item.code === code && item.type === type)
    );
    this.marketQuotes = this.marketQuotes.filter((item: Quote) =>
    !(item.code === code && item.type === type)
    );

    // 如果删的是当前选中标的，顺带清空 K 线
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
    const code: string = this.searchCode.trim()

    if (code.length !== 6) {
      this.searchError = '请输入 6 位代码'
      return
    }

    this.searchLoading = true
    this.searchError = ''

    const httpRequest = http.createHttp()
    try {
      const type: 'index' | 'stock' = this.currentMode
      const url: string = `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      })

      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendQuote = JSON.parse(res.result as string) as BackendQuote
        const newItem: Quote = this.mapBackendQuote(backendItem, type)

        this.mergeQuotesToCache([newItem])

        // 只保留当前模式 + 去重 + 新结果置顶
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

      if (res.responseCode === 404) {
        this.searchError = '未找到该代码'
      } else {
        this.searchError = `HTTP ${res.responseCode}`
      }
    } catch (e) {
      this.searchError = `${e}`
    } finally {
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

  private async loadGroupsFromPrefs(): Promise<void> {
    this.dbg('loadGroupsFromPrefs:enter')
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE)
    const jsonStr: string = (await pref.get(this.PREF_KEY_GROUPS, '')) as string

    if (jsonStr.trim().length === 0) {
      const g: StockGroup = { id: 1, name: '默认分组', stocks: [], createdAt: Date.now(), version: 0 }
      this.groups = [g]
      this.activeGroupId = 1
      this.nextGroupId = 2
      await this.saveGroupsToPrefs()

      this.syncActiveStocks() // ✅ 放这里（默认分组分支）
      this.dbg('loadGroupsFromPrefs:initDefault')
      return
    }

    // ✅ parsed 必须先声明
    const parsed: StockGroup[] = this.safeParseGroups(jsonStr)
    this.groups = parsed

    let maxId: number = 0
    for (let i: number = 0; i < parsed.length; i++) {
      if (parsed[i].id > maxId) maxId = parsed[i].id
    }
    this.nextGroupId = maxId + 1

    this.activeGroupId = parsed.length > 0 ? parsed[0].id : -1

    this.syncActiveStocks() // ✅ 放这里（parsed 分支，且在 activeGroupId 之后）
    this.dbg('loadGroupsFromPrefs:done')
  }


  private async saveGroupsToPrefs(): Promise<void> {
    const pref = await preferences.getPreferences(getContext(this), this.PREF_FILE);
    const jsonStr: string = JSON.stringify(this.groups);
    await pref.put(this.PREF_KEY_GROUPS, jsonStr);
    await pref.flush();
  }

  // 不用 any/unknown：对结构做兜底校验
  private safeParseGroups(jsonStr: string): StockGroup[] {
    try {
      const rawObj: Object = JSON.parse(jsonStr) as Object;
      if (!Array.isArray(rawObj)) {
        return [];
      }

      const arr: Object[] = rawObj as Object[];
      const result: StockGroup[] = [];

      for (let i: number = 0; i < arr.length; i++) {
        const item: Record<string, Object> = arr[i] as Record<string, Object>;

        const id: number = Number(item['id']);
        const name: string = String(item['name'] ?? '');
        const createdAt: number = Number(item['createdAt'] ?? Date.now());
        const version: number = Number(item['version'] ?? 0); // ✅ 老数据默认 0

        const stocksRaw: Object = item['stocks'] ?? [];
        const stocks: StockRef[] = [];

        if (Array.isArray(stocksRaw)) {
          const sr: Object[] = stocksRaw as Object[];
          for (let j: number = 0; j < sr.length; j++) {
            const s: Record<string, Object> = sr[j] as Record<string, Object>;
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

        if (!Number.isNaN(id) && name.length > 0) {
          result.push({ id, name, stocks, createdAt, version });
        }
      }
      return result;
    } catch (e) {
      return [];
    }
  }

  private findGroupIndex(groupId: number): number {
    for (let i: number = 0; i < this.groups.length; i++) {
      if (this.groups[i].id === groupId) {
        return i;
      }
    }
    return -1;
  }

  private async createGroup(name: string): Promise<void> {
    const n: string = name.trim();
    if (n.length === 0) return;

    const g: StockGroup = { id: this.nextGroupId, name: n, stocks: [], createdAt: Date.now(), version: 0 };
    this.nextGroupId += 1;

    const copy: StockGroup[] = [g, ...this.groups];
    this.groups = copy;
    this.activeGroupId = g.id;

    // ✅ 关键：立刻让 UI 的 active 链路同步（不需要再点一次）
    this.syncActiveStocks();           // 列表 & 股票数立刻正确
    this.refreshActiveGroupStats();    // 统计卡立刻正确（空组就是 0）

    await this.saveGroupsToPrefs();

    // ✅ 可选：新组为空也没关系，但保持流程一致
    await this.preloadActiveGroupQuotes();
    this.refreshActiveGroupStats();
  }


  private async deleteGroup(groupId: number): Promise<void> {
    const copy: StockGroup[] = [];
    for (let i: number = 0; i < this.groups.length; i++) {
      if (this.groups[i].id !== groupId) copy.push(this.groups[i]);
    }
    this.groups = copy;

    if (this.activeGroupId === groupId) {
      this.activeGroupId = this.groups.length > 0 ? this.groups[0].id : -1;
    }

    // ✅ 关键：删除导致 activeGroupId 变化时，立刻同步 active 链路
    this.syncActiveStocks();
    this.refreshActiveGroupStats();

    await this.saveGroupsToPrefs();

    // ✅ 如果还有分组：预热新 active 分组，然后再刷新统计
    if (this.activeGroupId >= 0) {
      await this.preloadActiveGroupQuotes();
      this.refreshActiveGroupStats();
    }
  }

  private async addStocksToGroup(groupId: number, codes: string[]): Promise<void> {
    this.dbg(`addStocksToGroup:enter gid=${groupId} codes=${JSON.stringify(codes)}`)

    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) {
      return
    }

    // 1) 收集已有 code，便于去重
    const existed: Set<string> = new Set<string>()
    const curStocks: StockRef[] = this.groups[idx].stocks
    for (let i: number = 0; i < curStocks.length; i++) {
      existed.add(curStocks[i].code)
    }

    // 2) 过滤 + 去重待添加 codes（同一批 codes 内也去重）
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
      addList.push({ code: code6 })
    }

    if (addList.length === 0) {
      this.dbg('addStocksToGroup:skip empty addList')
      return
    }

    // 3) 更新 groups（必须新数组 + 新对象，触发 UI/状态）
    const newGroups: StockGroup[] = this.groups.slice()
    const g: StockGroup = newGroups[idx]
    const newStocks: StockRef[] = g.stocks.concat(addList)
    newGroups[idx] = this.cloneGroupWithStocks(g, newStocks)
    this.groups = newGroups

    // 4) 如果是当前分组：立刻同步 activeStocks（让列表/统计立即可见）
    if (this.activeGroupId === groupId) {
      this.syncActiveStocks()
    }

    // 5) 业务提示（可选）
    this.watchlistMsg = `已添加 ${addList.length} 只，当前 ${newGroups[idx].stocks.length} 只`

    // 6) 持久化
    await this.saveGroupsToPrefs()
    this.dbg('addStocksToGroup:afterSavePrefs')

    // 7) 如果是当前分组：预热行情（行情回写 quoteCache 后统计会自动更新）
    if (this.activeGroupId === groupId) {
      await this.preloadActiveGroupQuotes()
      this.refreshActiveGroupStats()
      this.dbg('addStocksToGroup:afterPreload')
    }
  }



  private async removeStocksFromGroup(groupId: number, codes: string[]): Promise<void> {
    this.dbg(`removeStocksFromGroup:enter gid=${groupId} codes=${JSON.stringify(codes)}`)

    const idx: number = this.findGroupIndex(groupId)
    if (idx < 0) {
      return
    }

    // 1) 需要移除的 code 集合（去重 + 过滤 6 位）
    const toRemove: Set<string> = new Set<string>()
    for (let i: number = 0; i < codes.length; i++) {
      const code6: string = codes[i].trim()
      if (code6.length === 6) {
        toRemove.add(code6)
      }
    }
    if (toRemove.size === 0) {
      this.dbg('removeStocksFromGroup:skip empty toRemove')
      return
    }

    // 2) 生成新 stocks
    const oldStocks: StockRef[] = this.groups[idx].stocks
    const newStocks: StockRef[] = []
    for (let i: number = 0; i < oldStocks.length; i++) {
      const code: string = oldStocks[i].code
      if (!toRemove.has(code)) {
        newStocks.push(oldStocks[i])
      }
    }

    // 如果没变化，直接返回
    if (newStocks.length === oldStocks.length) {
      this.dbg('removeStocksFromGroup:skip no changes')
      return
    }

    // 3) 更新 groups（新数组 + 新对象）
    const newGroups: StockGroup[] = this.groups.slice()
    const g: StockGroup = newGroups[idx]
    newGroups[idx] = this.cloneGroupWithStocks(g, newStocks)
    this.groups = newGroups

    // 4) 如果是当前分组：立刻同步 activeStocks（列表/统计立即变）
    if (this.activeGroupId === groupId) {
      this.syncActiveStocks()
    }

    // 5) 提示（可选）
    this.watchlistMsg = `已删除 ${oldStocks.length - newStocks.length} 只，当前 ${newGroups[idx].stocks.length} 只`

    // 6) 持久化
    await this.saveGroupsToPrefs()
    this.dbg('removeStocksFromGroup:afterSavePrefs')

    // 7) 如果是当前分组：预热行情（可选，但保持一致性；也能补齐 cache）
    if (this.activeGroupId === groupId) {
      await this.preloadActiveGroupQuotes()
      this.refreshActiveGroupStats()
      this.dbg('removeStocksFromGroup:afterPreload')
    }
  }




  private cloneGroupWithStocks(g: StockGroup, stocks: StockRef[]): StockGroup {
    return {
      id: g.id,
      name: g.name,
      stocks: stocks,
      createdAt: g.createdAt,
      version: g.version + 1 // ✅ 关键：每次 stocks 变化都递增
    };
  }


  private getActiveGroup(): StockGroup | null {
    if (this.activeGroupId < 0) {
      return null;
    }
    const idx: number = this.findGroupIndex(this.activeGroupId);
    if (idx < 0) {
      return null;
    }
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

        if (!Number.isNaN(id) && stockCode.length === 6 && stockName.length > 0 && conditionText.length > 0) {
          out.push({ id, stockCode, stockName, conditionText, enabled })
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
      if (r.stockCode === code && r.conditionText === cond) {
        this.ruleMsg = '该规则已存在'
        return
      }
    }

    // 尽量从缓存/后端拿 name（你的 ensureQuoteCachedForCode 已有）
    await this.ensureQuoteCachedForCode(code)
    const q: Quote | null = this.findStockQuoteInCache(code)
    const name: string = (q !== null) ? q.name : code

    const newRule: NotifyRule = {
      id: this.nextRuleId,
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

      Button('发送一条测试通知')
        .margin({ left: 16, right: 16, top: 8, bottom: 8 })
        .onClick(async () => {
          const rule: NotifyRule = {
            id: 999,
            stockCode: '300750',
            stockName: '宁德时代',
            conditionText: '价格 ≥ 220.00',
            enabled: true
          }
          await this.triggerRuleAlert(rule)
        })
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
                      .onClick(() => {
                        this.onKlineSelected(idx);
                      })
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
        Blank()
        Text('命中后：系统通知 + 震动（真机）')
          .fontSize(10).fontColor('#777777')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })

      // ✅ 新增规则区
      Column({ space: 8 }) {
        Row({ space: 8 }) {
          TextInput({ placeholder: '股票代码(6位)', text: this.ruleStockCodeInput })
            .width('38%').height(32).fontSize(13)
            .backgroundColor(Color.White).fontColor(Color.Black)
            .onChange((v: string) => { this.ruleStockCodeInput = v.trim() })

          // 操作符：用两个小按钮最稳，不用复杂 Picker
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

          TextInput({ placeholder: '阈值', text: this.rulePriceInput })
            .width('24%').height(32).fontSize(13)
            .backgroundColor(Color.White).fontColor(Color.Black)
            .onChange((v: string) => { this.rulePriceInput = v.trim() })

          Button('添加')
            .height(32)
            .onClick(async () => { await this.addNotifyRuleFromUI() })
        }

        if (this.ruleMsg.length > 0) {
          Text(this.ruleMsg).fontSize(10).fontColor('#777777')
        }
      }
      .padding({ left: 16, right: 16, bottom: 10 })

      List() {
        ForEach(this.notifyRules, (rule: NotifyRule) => {
          this.buildNotifyRuleItem(rule);
        }, (rule: NotifyRule) => rule.id.toString())
      }
      .key(`notifyRules-${this.notifyRulesVer}`)  // 使用 notifyRulesVer 作为 key，确保每次更新时渲染

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
          Text(`${rule.stockName}（${rule.stockCode}）`)
            .fontSize(13).fontColor(Color.White)

          Text(rule.conditionText)
            .fontSize(11).fontColor('#BBBBBB')
        }.layoutWeight(1)

        Button('删除')
          .height(28)
          .padding({ left: 10, right: 10 })
          .backgroundColor('#1A1A1A')
          .fontColor('#FF7875')
          .borderRadius(10)
          .onClick(async () => {
            await this.deleteRule(rule.id);
          })
      }
      .padding({ left: 16, right: 16, top: 10, bottom: 10 })
      .backgroundColor('#101010')
    }
    .margin({ bottom: 1 })
  }

}
