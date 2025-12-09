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
  async fetchMarketIndices() {
    // ✅ 用和 UI 一致的状态
    this.indicesLoading = true;
    this.indicesError = '';

    const httpRequest = http.createHttp();
    try {
      const url = 'http://101.43.185.73:8000/api/indices/realtime';

      const res = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 5000,
        readTimeout: 5000,
      });

      console.info('== indices http response ==', JSON.stringify(res));

      if (res.responseCode === 200 && res.result) {
        const raw = JSON.parse(res.result as string) as BackendStockIndex[];
        console.info('== indices parsed ==', JSON.stringify(raw));

        this.marketIndices = raw.map((it: BackendStockIndex): StockIndex => {
          return {
            code: it.code,
            name: it.name,
            price: it.price,
            change: it.change,
            changePercent: it.change_percent,
          };
        });

        console.info('== indices mapped ==', JSON.stringify(this.marketIndices));
      } else {
        this.indicesError = `HTTP ${res.responseCode}`;
      }
    } catch (e) {
      // 这里之前你直接 `${e}`，很多时候就是 [Object object]
      console.error('== indices http error ==', JSON.stringify(e));
      this.indicesError = JSON.stringify(e); // 直接转成字符串
    } finally {
      httpRequest.destroy();
      this.indicesLoading = false;
    }
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
      // ★ 新增：指数代码查询区域
      this.buildIndexSearchArea();

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
  }


  // ===== 指数横向 strip，带加载/错误状态 =====
  @Builder
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
      .padding({ left: 16, right: 16, top: 8, bottom: 4 })

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
  }





  @Builder
  private buildKlineCard() {
    Column() {
      // 标题 + 周期切换
      Row({ space: 12 }) {
        Text('上证指数 K 线')
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

      // K 线图占位区域
      Column() {
        Text('K 线图区域（UI 占位）')
          .fontSize(12)
          .fontColor('#666666')
          .margin({ bottom: 8 })

        Row({ space: 6 }) {
          ForEach([1, 2, 3, 4, 5, 6, 7], (id: number) => {
            Column() {
              Rect().width(2).height(40).fill('#444444')
              Rect().width(8).height(24).fill(id % 2 === 0 ? '#39D98A' : '#FF4D4F')
              Rect().width(2).height(40).fill('#444444')
            }
          }, (id: number) => id.toString())
        }
      }
      .width('100%')
      .height(180)
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
        this.klinePeriodIndex = index
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
