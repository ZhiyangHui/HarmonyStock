// entry/src/main/ets/pages/Index.ets
import { TabBarOptions } from '@kit.ArkUI';
import http from '@ohos.net.http';

// ====== 一些数据结构定义 ======
interface StockIndex {
  code: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

interface IndexKlineGeom {
  upperWick: number;   // 上影线高度
  bodyHeight: number;  // 实体高度
  lowerWick: number;   // 下影线高度
  topGap: number;      // 整根蜡烛距离顶部的空白，用来“抬高 / 放低”
  isUp: boolean;       // 是否上涨（close >= open）
}


interface BackendStockIndex {
  code: string;
  name: string;
  price: number;
  change: number;
  change_percent: number;
}

interface StockQuote {
  name: string;
  code: string;
  price: number;
  change: number;
  changePercent: number;
}

interface StockGroup {
  id: number;
  name: string;
  stocks: StockQuote[];
}

interface NotifyRule {
  id: number;
  stockCode: string;
  stockName: string;
  conditionText: string; // 比如 "价格 ≥ 25.00"
  enabled: boolean;
}

interface IndexKlinePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}


interface KLinePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface BackendKLinePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}


// ====== 页面组件 ======
@Entry
@Component
struct HarmonyStockPage {
  // 顶部 Tab：0 市场 1 自选 2 通知
  @State currentTabIndex: number = 0;

  // ===== 指数数据：从后端拉取 =====
  @State marketIndices: StockIndex[] = [];
  @State indicesLoading: boolean = false;
  @State indicesError: string = '';

  // ===== 指数代码查询相关状态 =====
  @State indexSearchCode: string = ''        // 输入框内容
  @State indexSearchLoading: boolean = false // 查询中
  @State indexSearchError: string = ''       // 查询错误
  @State searchedIndices: StockIndex[] = []  // ✅ 所有搜索出来的指数

  // 当前 K 线周期：0 日K 1 周K 2 月K
  @State klinePeriodIndex: number = 0;

  // 当前选中的指数，用来切换 K 线
  @State selectedIndexCode: string = '';
  @State selectedIndexName: string = '';

  // 指数 K 线的数据和状态
  @State klineLoading: boolean = false;
  @State klineError: string = '';
  @State klinePoints: IndexKlinePoint[] = [];

  // ⭐ 新增：专门给坐标轴用的状态
  @State klinePriceTicks: number[] = [];   // y 轴价格刻度
  @State klineDateLabels: string[] = [];   // x 轴日期标签（3 个点）

  // ⭐ 新增：K 线几何信息 & 轴刻度
  @State klineGeoms: IndexKlineGeom[] = [];
  @State klineMinPrice: number = 0;
  @State klineMaxPrice: number = 0;
  @State klineXLabels: string[] = [];  // 三个日期标签

  // ⭐ 当前选中的 K 线（用于显示详细数据 + 高亮）
  @State selectedKlineIndex: number = -1;
  @State selectedKlineDate: string = '';
  @State selectedKlineOpen: number = 0;
  @State selectedKlineHigh: number = 0;
  @State selectedKlineLow: number = 0;
  @State selectedKlineClose: number = 0;

  // ⭐ 原始日 K 线（后端返回的原始数据）
  @State rawDailyKline: IndexKlinePoint[] = [];

  // 每根 K 线实体宽度
  private candleBodyWidth: number = 8;
  // 相邻两根 K 线之间的间距（Row 的 space）
  private candleGap: number = 4;
  // 右侧额外预留一点空白，避免最后一个日期被卡边上
  private klineTailPadding: number = 16;




  // ===== 下面这些还是你原来的自选股 / 通知假数据，不动 =====
  private groups: StockGroup[] = [
    {
      id: 1,
      name: '科技股',
      stocks: [
        { name: '宁德时代', code: '300750', price: 210.35, change: -1.25, changePercent: -0.59 },
        { name: '中芯国际', code: '688981', price: 58.20, change: 0.60, changePercent: 1.04 }
      ]
    },
    {
      id: 2,
      name: '新能源',
      stocks: [
        { name: '隆基绿能', code: '601012', price: 32.10, change: 0.35, changePercent: 1.10 }
      ]
    }
  ];

  private notifyRules: NotifyRule[] = [
    {
      id: 1,
      stockCode: '300750',
      stockName: '宁德时代',
      conditionText: '价格 ≥ 220.00',
      enabled: true
    },
    {
      id: 2,
      stockCode: '601012',
      stockName: '隆基绿能',
      conditionText: '价格 ≤ 28.00',
      enabled: false
    }
  ];



  // ===== 页面生命周期：进来时拉一次指数 =====
  aboutToAppear() {
    this.fetchMarketIndices();
  }

  // ===== 根据代码查询单个指数 =====
  private async queryIndexByCode() {
    const code: string = this.indexSearchCode;

    if (!code || code.length !== 6) {
      this.indexSearchError = '请输入 6 位指数代码';
      return;
    }

    this.indexSearchLoading = true;
    this.indexSearchError = '';

    const httpRequest = http.createHttp();
    try {
      const url = `http://101.43.185.73:8000/api/index/by_code?code=${code}`;

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 5000,
        readTimeout: 5000,
      });

      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendStockIndex =
          JSON.parse(res.result as string) as BackendStockIndex;

        const newItem: StockIndex = {
          code: backendItem.code,
          name: backendItem.name,
          price: backendItem.price,
          change: backendItem.change,
          changePercent: backendItem.change_percent,
        };

        // 1）先看 strip 里是否已经有这个 code（避免重复）
        const existsInSearched: boolean = this.searchedIndices.some(
          (it: StockIndex) => it.code === newItem.code
        );
        const existsInMarket: boolean = this.marketIndices.some(
          (it: StockIndex) => it.code === newItem.code
        );

        if (!existsInSearched && !existsInMarket) {
          // 2）插到 searchedIndices 的开头
          const copy: StockIndex[] = [newItem];
          for (let i = 0; i < this.searchedIndices.length; i++) {
            copy.push(this.searchedIndices[i]);
          }
          this.searchedIndices = copy;
        }
      } else if (res.responseCode === 404) {
        this.indexSearchError = '未找到该指数代码';
      } else {
        this.indexSearchError = `HTTP ${res.responseCode}`;
      }
    } catch (e) {
      this.indexSearchError = `${e}`;
    } finally {
      httpRequest.destroy();
      this.indexSearchLoading = false;
    }
  }

  // 删除某个指数卡片（同时从搜索结果和原始指数里删）
  private removeIndexCard(code: string) {
    // 删搜索出来的那几张卡片
    this.searchedIndices = this.searchedIndices.filter((item: StockIndex) => item.code !== code);
    // 如果你也想从原始列表中一起删，就保留下面这行；
    // 如果只想删“前面搜索出来的卡片”，可以把下面这一行删掉
    this.marketIndices = this.marketIndices.filter((item: StockIndex) => item.code !== code);
  }


  // ===== 向 FastAPI 后端请求指数数据 =====
  async fetchMarketIndices(): Promise<void> {
    this.indicesLoading = true;
    this.indicesError = '';

    const httpRequest = http.createHttp();
    try {
      const url: string = 'http://101.43.185.73:8000/api/indices/realtime';

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 5000,
        readTimeout: 5000,
      });

      console.info('== indices http response ==', JSON.stringify(res));

      if (res.responseCode === 200 && res.result) {
        const raw: BackendStockIndex[] = JSON.parse(res.result as string) as BackendStockIndex[];
        console.info('== indices parsed ==', JSON.stringify(raw));

        const mapped: StockIndex[] = raw.map((it: BackendStockIndex): StockIndex => {
          const item: StockIndex = {
            code: it.code,
            name: it.name,
            price: it.price,
            change: it.change,
            changePercent: it.change_percent,
          };
          return item;
        });

        this.marketIndices = mapped;
        console.info('== indices mapped ==', JSON.stringify(this.marketIndices));

        // ⭐ 新增：如果还没有选中的指数，就默认选 strip 里的第一个
        if (this.selectedIndexCode === '') {
          const list: StockIndex[] = this.getIndexStripList();
          if (list.length > 0) {
            const first: StockIndex = list[0];
            this.selectedIndexCode = first.code;
            this.selectedIndexName = first.name;

            const period: string = this.getKlinePeriodParam();
            this.fetchIndexKline(first.code, period);
          }

        }

      } else {
        this.indicesError = `HTTP ${res.responseCode}`;
        console.error('== indices http non-200 ==', JSON.stringify(res));
      }
    } catch (e) {
      const msg: string = JSON.stringify(e);
      console.error('== indices http error ==', msg);
      this.indicesError = msg;
    } finally {
      httpRequest.destroy();
      this.indicesLoading = false;
    }
  }


  // 根据 this.klinePoints 计算几何信息 & 轴刻度
  private recomputeKlineGeom(): void {
    const points: IndexKlinePoint[] = this.klinePoints;
    const geoms: IndexKlineGeom[] = [];
    const labels: string[] = [];

    if (points.length === 0) {
      this.klineGeoms = geoms;
      this.klineXLabels = labels;
      return;
    }

    // 1. 找最高价 / 最低价
    let minPrice: number = points[0].low;
    let maxPrice: number = points[0].high;
    for (let i: number = 1; i < points.length; i++) {
      const p: IndexKlinePoint = points[i];
      if (p.low < minPrice) {
        minPrice = p.low;
      }
      if (p.high > maxPrice) {
        maxPrice = p.high;
      }
    }
    if (maxPrice <= minPrice) {
      maxPrice = minPrice + 1.0;
    }
    this.klineMinPrice = minPrice;
    this.klineMaxPrice = maxPrice;

    const range: number = maxPrice - minPrice;

    // 整个绘图区高度
    const fullHeight: number = 120;
    const maxBodyHeight: number = 60;
    const maxWickHeight: number = 40;

    // 2. 逐个点算出蜡烛的几何信息
    for (let i: number = 0; i < points.length; i++) {
      const pt: IndexKlinePoint = points[i];

      // 上下影线总高度
      const span: number = pt.high - pt.low;
      let wickTotal: number = span / range * maxWickHeight;
      if (wickTotal < 4) {
        wickTotal = 4;
      }
      const upperWick: number = wickTotal / 2.0;
      const lowerWick: number = wickTotal / 2.0;

      // 实体高度
      const diff: number = Math.abs(pt.close - pt.open);
      let bodyHeight: number = diff / range * maxBodyHeight;
      if (bodyHeight < 4) {
        bodyHeight = 4;
      }

      // 这根蜡烛整体在竖直方向的位置：用 (高+低)/2 做一个“中点价”
      const midPrice: number = (pt.high + pt.low) / 2.0;
      const ratio: number = (maxPrice - midPrice) / range;  // 0~1，越大越靠上

      // 顶部空白 = 比例 * 可用高度
      let topGap: number = ratio * (fullHeight - wickTotal - bodyHeight);
      if (topGap < 0) {
        topGap = 0;
      }

      const geom: IndexKlineGeom = {
        upperWick: upperWick,
        bodyHeight: bodyHeight,
        lowerWick: lowerWick,
        topGap: topGap,
        isUp: pt.close >= pt.open,
      };
      geoms.push(geom);
    }

    // 3. x 轴标签：最多取 5 个点，均匀分布在头到尾之间
    const total: number = points.length;
    const maxLabels: number = 5;
    const actualLabels: number = total >= maxLabels ? maxLabels : total;

    if (actualLabels > 0) {
      const lastIndex: number = total - 1;

      for (let i: number = 0; i < actualLabels; i++) {
        // 均匀取样：i 从 0 到 actualLabels-1，映射到 0~lastIndex
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


  // ===== 根据当前选中的指数，向后端请求 K 线数据 =====
  // 根据当前选中的指数 + 周期，向后端请求 K 线数据
  async fetchIndexKline(code: string, period: string): Promise<void> {
    this.klineLoading = true;
    this.klineError = '';

    const httpRequest = http.createHttp();
    try {
      const url: string =
        `http://101.43.185.73:8000/api/indices/kline?code=${code}&period=${period}&limit=60`;

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 5000,
        readTimeout: 5000,
      });

      if (res.responseCode === 200 && res.result) {
        const raw: IndexKlinePoint[] =
          JSON.parse(res.result as string) as IndexKlinePoint[];
        this.klinePoints = raw;

        // ⭐ 所有几何计算都在这里一次性完成
        this.recomputeKlineGeom();
      } else {
        this.klineError = `HTTP ${res.responseCode}`;
        this.klinePoints = [];
        this.klineGeoms = [];
        this.klineXLabels = [];
      }
    } catch (e) {
      this.klineError = `${e}`;
      this.klinePoints = [];
      this.klineGeoms = [];
      this.klineXLabels = [];
    } finally {
      httpRequest.destroy();
      this.klineLoading = false;
    }
  }


  private getKlinePeriodParam(): string {
    if (this.klinePeriodIndex === 1) {
      return 'week';   // 周 K
    }
    if (this.klinePeriodIndex === 2) {
      return 'month';  // 月 K
    }
    return 'day';      // 默认日 K
  }

  // 根据当前 klinePeriodIndex（日/周/月），从原始日K生成要显示的 klinePoints
  private recomputeKlineView(): void {
    let viewPoints: IndexKlinePoint[] = [];

    if (this.rawDailyKline.length === 0) {
      this.klinePoints = [];
      this.klineGeoms = [];
      this.klineXLabels = [];
      return;
    }

    // 0: 日K  1: 周K  2: 月K
    if (this.klinePeriodIndex === 0) {
      // 日K：直接用原始数据
      viewPoints = this.rawDailyKline;
    } else if (this.klinePeriodIndex === 1) {
      // 周K：用前端把日K按 5 天一组粗略合成
      viewPoints = this.aggregateToWeek(this.rawDailyKline);
    } else {
      // 月K：按月份合成
      viewPoints = this.aggregateToMonth(this.rawDailyKline);
    }

    this.klinePoints = viewPoints;
    // ⭐ 用新的 klinePoints 重新算几何信息和坐标轴
    this.recomputeKlineGeom();
  }

  // 粗略按“每 5 个交易日”合成一根周K
  private aggregateToWeek(points: IndexKlinePoint[]): IndexKlinePoint[] {
    const result: IndexKlinePoint[] = [];
    const n: number = points.length;
    let i: number = 0;

    while (i < n) {
      const chunk: IndexKlinePoint[] = [];
      for (let j: number = 0; j < 5 && i < n; j++, i++) {
        chunk.push(points[i]);
      }
      if (chunk.length === 0) {
        break;
      }

      const first: IndexKlinePoint = chunk[0];
      const last: IndexKlinePoint = chunk[chunk.length - 1];

      let high: number = first.high;
      let low: number = first.low;
      let volume: number = 0;

      for (let k: number = 0; k < chunk.length; k++) {
        const p: IndexKlinePoint = chunk[k];
        if (p.high > high) {
          high = p.high;
        }
        if (p.low < low) {
          low = p.low;
        }
        volume += p.volume;
      }

      const agg: IndexKlinePoint = {
        date: last.date,          // 用这一周最后一天的日期
        open: first.open,         // 第一日开盘
        close: last.close,        // 最后一日收盘
        high: high,
        low: low,
        volume: volume,
      };
      result.push(agg);
    }

    return result;
  }

  // 按月份合成月K（根据 date 的 "YYYY-MM" 粗略分组）
  private aggregateToMonth(points: IndexKlinePoint[]): IndexKlinePoint[] {
    const result: IndexKlinePoint[] = [];
    let currentKey: string = '';
    let monthOpen: number = 0;
    let monthHigh: number = 0;
    let monthLow: number = 0;
    let monthVolume: number = 0;
    let hasData: boolean = false;

    for (let i: number = 0; i < points.length; i++) {
      const p: IndexKlinePoint = points[i];
      const key: string = p.date.length >= 7 ? p.date.substring(0, 7) : p.date; // "YYYY-MM"

      if (!hasData) {
        // 第一条
        currentKey = key;
        monthOpen = p.open;
        monthHigh = p.high;
        monthLow = p.low;
        monthVolume = p.volume;
        hasData = true;
      } else if (key === currentKey) {
        // 同一个月内，累积 high / low / volume
        if (p.high > monthHigh) {
          monthHigh = p.high;
        }
        if (p.low < monthLow) {
          monthLow = p.low;
        }
        monthVolume += p.volume;
      } else {
        // 月份切换，把上一个月收尾
        const prev: IndexKlinePoint = points[i - 1];
        result.push({
          date: prev.date,       // 上一个月最后一天的日期
          open: monthOpen,
          close: prev.close,
          high: monthHigh,
          low: monthLow,
          volume: monthVolume,
        });

        // 开始下一个月
        currentKey = key;
        monthOpen = p.open;
        monthHigh = p.high;
        monthLow = p.low;
        monthVolume = p.volume;
      }
    }

    // 最后一个月补上
    if (hasData) {
      const last: IndexKlinePoint = points[points.length - 1];
      result.push({
        date: last.date,
        open: monthOpen,
        close: last.close,
        high: monthHigh,
        low: monthLow,
        volume: monthVolume,
      });
    }

    return result;
  }


  private onKlineSelected(index: number): void {
    if (index < 0 || index >= this.klinePoints.length) {
      return;
    }

    const pt: IndexKlinePoint = this.klinePoints[index];

    this.selectedKlineIndex = index;
    this.selectedKlineDate = pt.date;
    this.selectedKlineOpen = pt.open;
    this.selectedKlineHigh = pt.high;
    this.selectedKlineLow = pt.low;
    this.selectedKlineClose = pt.close;
  }




  private getIndexStripList(): StockIndex[] {
    const result: StockIndex[] = [];
    // 先把查出来的放前面
    for (let i = 0; i < this.searchedIndices.length; i++) {
      result.push(this.searchedIndices[i]);
    }
    // 再把原始指数列表放后面
    for (let j = 0; j < this.marketIndices.length; j++) {
      result.push(this.marketIndices[j]);
    }
    return result;
  }

  // 计算整段 K 线区域的总宽度，给蜡烛 Row 和 x 轴 Row 共用
  // 计算整段 K 线区域的总宽度，刚好包住所有蜡烛
  private getKlineTotalWidth(): number {
    const count: number = this.klineGeoms.length;
    if (count <= 0) {
      return 0;
    }
    // 所有实体宽度 + 间距（有 count - 1 个 gap）
    return count * this.candleBodyWidth + (count - 1) * this.candleGap;
  }


  // 自定义 TabBar：这里就能完全控制文字颜色
  @Builder tabBarBuilder(title: string, index: number) {
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

  // ====== Tab 1：市场（指数 + 大盘 K 线 + 桌面卡片预览） ======
  @Builder
  private buildMarketTab() {
    Column() {
      Row(){
        Text('指数卡片')
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Color.White)
          .margin(15)

        // ★ 新增：指数代码查询区域
        this.buildIndexSearchArea();
      }
      .width('100%')                             // ⭐ 拉满整行宽度
      .justifyContent(FlexAlign.SpaceBetween)    // ⭐ 左右两端对齐
      .alignItems(VerticalAlign.Center)

      // 指数横向滚动
      this.buildIndexStrip();

      // 大盘 K 线卡片
      this.buildKlineCard();

      // 桌面行情卡片预览
      this.buildDesktopCardPreview();

    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    .alignItems(HorizontalAlign.Start)
  }


  // ===== 指数横向 strip，带加载/错误状态 + 可删除的搜索卡片 =====
  @Builder
  private buildIndexStrip() {
    if (this.indicesLoading) {
      Row() {
        Text('指数加载中…')
          .fontSize(12)
          .fontColor('#888888')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    } else if (this.indicesError) {
      Row() {
        Text(`指数加载失败：${this.indicesError}`)
          .fontSize(12)
          .fontColor('#FF7875')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    } else {
      // ✅ 正常数据：先渲染“搜索结果卡片”，再渲染“后台实时卡片”
      Scroll() {
        Row({ space: 12 }) {
          // 搜索出来的卡片
          ForEach(this.searchedIndices, (item: StockIndex) => {
            this.buildIndexCard(item);
          }, (item: StockIndex) => `search-${item.code}`)

          // 后端实时卡片
          ForEach(this.marketIndices, (item: StockIndex) => {
            this.buildIndexCard(item);
          }, (item: StockIndex) => `rt-${item.code}`)
        }
        .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      }
      .scrollable(ScrollDirection.Horizontal)
    }
  }



  // ===== 指数代码查询区域（输入 + 结果展示） =====
  @Builder
  private buildIndexSearchArea() {
    Column() {
      // 上：输入 + 按钮
      Row({ space: 8 }) {
        TextInput({ placeholder: '输入指数代码，例如 000001' })
          .width('60%')
          .height(32)
          .fontSize(14)
          .backgroundColor(Color.White)  // ✅ 白色输入框
          .fontColor(Color.Black)
          .onChange((value: string) => {
            this.indexSearchCode = value.trim();
          })

        Button('查询')
          .height(32)
          .onClick(() => {
            this.queryIndexByCode();
          })
      }
      .padding({ left: 16, right: 48, top: 8, bottom: 4 })

      // 下：状态提示
      if (this.indexSearchLoading) {
        Row() {
          Text('正在查询指数…')
            .fontSize(12)
            .fontColor('#888888')
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 8 })
      } else if (this.indexSearchError) {
        Row() {
          Text(`查询失败：${this.indexSearchError}`)
            .fontSize(12)
            .fontColor('#FF7875')
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 8 })
      }
    }
  }


  @Builder
  private buildIndexCard(item: StockIndex) {
    Column() {
      // 顶部一行：名称 + 右上角 X
      Row() {
        Text(item.name)
          .fontSize(12)
          .fontColor('#AAAAAA')

        Text('✕')
          .fontSize(12)
          .fontColor('#777777')
          .onClick(() => {
            // 直接删掉这个 code 对应的卡片
            this.removeIndexCard(item.code);
          })
      }
      .width('100%')                             // ⭐ 拉满整行宽度
      .justifyContent(FlexAlign.SpaceBetween)    // ⭐ 左右两端对齐
      .alignItems(VerticalAlign.Center)

      // 价格
      Text(item.price.toFixed(2))
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor(item.change >= 0 ? '#FF4D4F' : '#39D98A')
        .margin({ top: 4 })

      // 涨跌
      Text(`${item.change.toFixed(2)}  ${item.changePercent.toFixed(2)}%`)
        .fontSize(12)
        .fontColor(item.change >= 0 ? '#FF7875' : '#6BE49B')
        .margin({ top: 4 })
    }
    .padding(12)
    .backgroundColor('#171717')
    .borderRadius(12)
    .width(140)
    .onClick(() => {
      // 点击整张卡片：切换当前选中的指数 + 请求 K 线
      this.selectedIndexCode = item.code;
      this.selectedIndexName = item.name;
      const period: string = this.getKlinePeriodParam();
      this.fetchIndexKline(item.code, period);
    })
  }


  @Builder
  private buildKlineCard() {
    Column() {
      // 标题 + 周期切换
      Row({ space: 12 }) {
        Text(this.selectedIndexName !== '' ? `${this.selectedIndexName} K 线` : '指数 K 线')
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

      // ⭐ 当前选中 K 线的详细信息条
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
            // 上：y 轴 + (K 线 + x 轴 Scroll)
            Row() {
              // y 轴刻度：最高 / 中间 / 最低（保持不动）
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
              .height(120)
              .justifyContent(FlexAlign.SpaceBetween)

              // 右侧：K 线 + x 轴 一起横向可滚动
              Scroll() {
                Column() {
                  // ① 蜡烛图区域
                  Row({ space: this.candleGap }) {
                    ForEach(this.klineGeoms, (g: IndexKlineGeom, idx: number) => {
                      Column() {
                        // 顶部空白：决定这根蜡烛整体的“高低位置”
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

                        // 底部补一点空白
                        Blank().height(4)
                      }
                      // ⭐ 高亮当前选中的那一根
                      .border({
                        width: this.selectedKlineIndex === idx ? 1 : 0,
                        color: this.selectedKlineIndex === idx ? '#FACC15' : '#000000'
                      })
                      // ⭐ 点哪根选哪根
                      .onClick(() => {
                        this.onKlineSelected(idx);
                      })
                    }, (g: IndexKlineGeom, idx: number) => `${idx}`)
                  }
                  .width(this.getKlineTotalWidth())   // K 线区域的总宽度
                  .height(120)

                  // ② x 轴日期标签：放在同一个 Scroll 里，宽度与 K 线一致
                  // ② x 轴日期标签：放在同一个 Scroll 里，宽度与 K 线一致
                  if (this.klineXLabels.length > 0) {
                    Row() {
                      ForEach(this.klineXLabels, (label: string, idx: number) => {
                        Text(label)
                          .fontSize(10)
                          .fontColor('#777777')
                      }, (label: string, idx: number) => `${idx}-${label}`)
                    }
                    .width(this.getKlineTotalWidth())
                    .justifyContent(FlexAlign.SpaceBetween) // N 个点：首、中、若干中间、尾
                    .margin({ bottom: 10 })
                  }

                }.padding({ right: 32})
              }
              .scrollable(ScrollDirection.Horizontal)
              .height(140)
              .margin({ left: 8 })
            }
            .width('100%')
          }
        }
      }
      .width('100%')
      .height(160)
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
        // 如果点的是当前已经选中的周期，就不用重复请求
        if (this.klinePeriodIndex === index) {
          return;
        }
        this.klinePeriodIndex = index;

        // 已经选中了某个指数，才有必要重新拉 K 线
        if (this.selectedIndexCode !== '') {
          const period: string = this.getKlinePeriodParam();
          this.fetchIndexKline(this.selectedIndexCode, period);
        }
      })
  }

  @Builder
  private buildDesktopCardPreview() {
    Column() {
      Row() {
        Text('桌面行情卡片预览')
          .fontSize(14)
          .fontColor('#CCCCCC')
        Blank()
        Text('仅 UI 展示')
          .fontSize(10)
          .fontColor('#666666')
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 8 })

      Row({ space: 12 }) {
        Column() {
          this.buildDesktopCardSmall();
        }

        Column() {
          this.buildDesktopCardLarge();
        }
      }
      .padding({ left: 16, right: 16, bottom: 16 })
    }
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
      Scroll() {
        Row({ space: 8 }) {
          ForEach(this.groups, (group: StockGroup) => {
            Column() {
              Text(group.name)
                .fontSize(12)
                .fontColor('#DDDDDD')
            }
            .padding({ left: 12, right: 12, top: 6, bottom: 6 })
            .backgroundColor('#161616')
            .borderRadius(12)
          }, (group: StockGroup) => group.id.toString())
        }
        .padding({ left: 16, right: 16, top: 12, bottom: 8 })
      }.scrollable(ScrollDirection.Horizontal)

      if (this.groups.length > 0) {
        Column() {
          Row() {
            Text(this.groups[0].name)
              .fontSize(14)
              .fontColor('#FFFFFF')
            Blank()
            Text('组内统计（占位）')
              .fontSize(11)
              .fontColor('#777777')
          }
          .width('100%')
          .padding({ left: 16, right: 16, bottom: 8 })

          List() {
            ForEach(this.groups[0].stocks, (s: StockQuote) => {
              this.buildWatchlistRow(s)
            }, (s: StockQuote) => s.code)
          }
        }
      } else {
        Column() {
          Text('暂无自选分组')
            .fontSize(12)
            .fontColor('#777777')
            .margin({ top: 32 })
        }
        .width('100%')
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
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
        Text('通知规则')
          .fontSize(14)
          .fontColor(Color.White)
        Blank()
        Text('满足条件时震铃 + 系统通知（占位）')
          .fontSize(10)
          .fontColor('#777777')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })

      List() {
        ForEach(this.notifyRules, (rule: NotifyRule) => {
          this.buildNotifyRuleItem(rule);
        }, (rule: NotifyRule) => rule.id.toString())
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }

  @Builder
  private buildNotifyRuleItem(rule: NotifyRule) {
    ListItem() {
      Row() {
        Column({ space: 4 }) {
          Text(`${rule.stockName}（${rule.stockCode}）`)
            .fontSize(13)
            .fontColor(Color.White)
          Text(rule.conditionText)
            .fontSize(11)
            .fontColor('#BBBBBB')
        }

        Blank()

        Text(rule.enabled ? '启用' : '停用')
          .fontSize(11)
          .fontColor(rule.enabled ? '#39D98A' : '#777777')
      }
      .padding({ left: 16, right: 16, top: 10, bottom: 10 })
      .backgroundColor('#101010')
    }
    .margin({ bottom: 1 })
  }
}
