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

  // 当前 K 线周期：0 日K 1 周K 2 月K
  @State klinePeriodIndex: number = 0;

  // ===== 页面生命周期：进来时拉一次指数 =====
  aboutToAppear() {
    this.fetchMarketIndices();
  }

  // ===== 向 FastAPI 后端请求指数数据 =====
  async fetchMarketIndices() {
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

      if (res.responseCode === 200 && res.result) {
        const raw = JSON.parse(res.result as string) as BackendStockIndex[];

        this.marketIndices = raw.map((it: BackendStockIndex): StockIndex => {
          const item: StockIndex = {
            code: it.code,
            name: it.name,
            price: it.price,
            change: it.change,
            changePercent: it.change_percent,
          };
          return item;
        });
      } else {
        this.indicesError = `HTTP ${res.responseCode}`;
      }
    } catch (e) {
      this.indicesError = `${e}`;
    } finally {
      httpRequest.destroy();
      this.indicesLoading = false;
    }
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
  private buildIndexStrip() {
    if (this.indicesLoading) {
      // 加载中
      Row() {
        Text('指数加载中…')
          .fontSize(12)
          .fontColor('#888888')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    } else if (this.indicesError) {
      // 加载失败
      Row() {
        Text(`指数加载失败：${this.indicesError}`)
          .fontSize(12)
          .fontColor('#FF7875')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    } else {
      // 正常数据
      Scroll() {
        Row({ space: 12 }) {
          ForEach(this.marketIndices, (item: StockIndex) => {
            this.buildIndexCard(item);
          }, (item: StockIndex) => item.code)
        }
        .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      }.scrollable(ScrollDirection.Horizontal)
    }
  }

  @Builder
  private buildIndexCard(item: StockIndex) {
    Column() {
      Text(item.name)
        .fontSize(12)
        .fontColor('#AAAAAA')

      Text(item.price.toFixed(2))
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor(item.change >= 0 ? '#FF4D4F' : '#39D98A')
        .margin({ top: 4 })

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
