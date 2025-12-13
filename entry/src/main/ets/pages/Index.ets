// entry/src/main/ets/pages/Index.ets
import { TabBarOptions } from '@kit.ArkUI';
import http from '@ohos.net.http';

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

  private readonly HTTP_CONNECT_TIMEOUT: number = 55000; // 35s
  private readonly HTTP_READ_TIMEOUT: number = 55000;    // 35s


  // ===== 自选 / 通知假数据（保持原样） =====
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
    // 默认先加载“指数模式”的行情
    this.fetchQuotes(this.currentMode);
  }

  // ===== 拉取行情列表（指数 / 股票共用） =====
  private async fetchQuotes(type: 'index' | 'stock'): Promise<void> {
    this.listLoading = true;
    this.listError = '';

    const httpRequest = http.createHttp();
    try {
      const url: string = `http://101.43.185.73:8000/api/quote?type=${type}`;

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      });


      if (res.responseCode === 200 && res.result) {
        const raw: BackendQuote[] =
          JSON.parse(res.result as string) as BackendQuote[];

        this.marketQuotes = raw.map((it: BackendQuote): Quote =>
        this.mapBackendQuote(it, type)
        );

        // 如果当前没有选中任何标的，则默认选第一只
        if (this.selectedQuoteCode === '' && this.marketQuotes.length > 0) {
          const first: Quote = this.marketQuotes[0];
          this.selectedQuoteCode = first.code;
          this.selectedQuoteName = first.name;
          this.selectedQuoteType = first.type;

          const period: string = this.getKlinePeriodParam();
          this.fetchKline(first.type, first.code, period);
        }
      } else {
        this.listError = `HTTP ${res.responseCode}`;
        this.marketQuotes = [];
      }
    } catch (e) {
      this.listError = `${e}`;
      this.marketQuotes = [];
    } finally {
      httpRequest.destroy();
      this.listLoading = false;
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
    const code: string = this.searchCode;

    if (!code || code.length !== 6) {
      this.searchError = '请输入 6 位代码';
      return;
    }

    this.searchLoading = true;
    this.searchError = '';

    const httpRequest = http.createHttp();
    try {
      const type: 'index' | 'stock' = this.currentMode;
      const url: string =
        `http://101.43.185.73:8000/api/quote/by_code?type=${type}&code=${code}`;

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: this.HTTP_CONNECT_TIMEOUT,
        readTimeout: this.HTTP_READ_TIMEOUT
      });


      if (res.responseCode === 200 && res.result) {
        const backendItem: BackendQuote =
          JSON.parse(res.result as string) as BackendQuote;

        const newItem: Quote = this.mapBackendQuote(backendItem, type);

        const existsInSearched: boolean = this.searchedQuotes.some(
          (it: Quote) => it.code === newItem.code && it.type === newItem.type
        );
        const existsInMarket: boolean = this.marketQuotes.some(
          (it: Quote) => it.code === newItem.code && it.type === newItem.type
        );

        if (!existsInSearched && !existsInMarket) {
          const copy: Quote[] = [newItem];
          for (let i: number = 0; i < this.searchedQuotes.length; i++) {
            copy.push(this.searchedQuotes[i]);
          }
          this.searchedQuotes = copy;
        }
      } else if (res.responseCode === 404) {
        this.searchError = '未找到该代码';
      } else {
        this.searchError = `HTTP ${res.responseCode}`;
      }
    } catch (e) {
      this.searchError = `${e}`;
    } finally {
      httpRequest.destroy();
      this.searchLoading = false;
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
      // 直接 inline 取 expanded，避免 Builder 内 const/let
      // expanded = this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false

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
              .onClick(() => {
                this.removeQuoteCard(item.code, item.type);
              })
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
        .onClick(() => {
          this.selectedQuoteCode = item.code;
          this.selectedQuoteName = item.name;
          this.selectedQuoteType = item.type;
          const period: string = this.getKlinePeriodParam();
          this.fetchKline(item.type, item.code, period);
        })

        // ===== 右下角 +/-：只负责展开收起 =====
        Row() {
          Blank()
          Text((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? '－' : '＋')
            .fontSize(18)
            .fontColor('#CCCCCC')
            .onClick(() => {
              this.toggleCardExpanded(item);
            })
        }
        .width('100%')
        .margin({ top: 8 })
      }
      .padding((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 20 : 12)
      .backgroundColor('#171717')
      .borderRadius(12)
      .width((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 200 : 140)
      .height((this.cardExpandedMap.get(`${item.type}-${item.code}`) ?? false) ? 150 : undefined)
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
