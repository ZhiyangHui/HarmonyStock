// 桌面卡片使用的本地存储容器
const storage = new LocalStorage()

// 路由到 App 时携带的参数结构
interface RouterParams {
  formId: string
}

// 桌面卡片触发跳转的动作配置
interface RouterActionOption {
  action: 'router'
  abilityName: string
  params: RouterParams
}

// 系统提供的卡片跳转方法（点击卡片进入 App）
declare function postCardAction(ctx: Object, option: RouterActionOption): void

// 桌面卡片支持的尺寸标识（由系统或 FormAbility 注入）
type FormDimensionText = '1*2' | '2*2' | '2*4'

// 桌面卡片入口组件，绑定 LocalStorage
@Entry(storage)
@Component
struct QuoteWidget {

  // 以下字段全部由 QuoteFormAbility.updateForm 写入 LocalStorage
  @LocalStorageProp('formId') formId: string = ''
  @LocalStorageProp('title') title: string = ''
  @LocalStorageProp('code') code: string = ''
  @LocalStorageProp('name') name: string = ''
  @LocalStorageProp('priceText') priceText: string = '--'
  @LocalStorageProp('changeText') changeText: string = '--'
  @LocalStorageProp('pctText') pctText: string = '--'
  @LocalStorageProp('tsText') tsText: string = ''

  // 当前卡片尺寸（由 QuoteFormAbility 在 updateForm 时同步）
  @LocalStorageProp('dimension') dimension: FormDimensionText = '2*2'

  // 点击卡片时跳转的目标 Ability
  readonly abilityName: string = 'EntryAbility'

  // 根据涨跌额文本判断当前是涨还是跌（用于颜色）
  private isUp(): boolean {
    if (this.changeText === '--') {
      return true
    }
    return !this.changeText.startsWith('-')
  }

  // 点击桌面卡片，携带 formId 跳转到 App
  private openApp(): void {
    postCardAction(this, {
      action: 'router',
      abilityName: this.abilityName,
      params: { formId: this.formId }
    })
  }

  build() {
    Column() {
      // 根据当前卡片尺寸选择不同布局
      this.buildByDimension()
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#151515')
    .borderRadius(16)
    .onClick(() => {
      this.openApp()
    })
  }

  // 根据 dimension 分发到不同 Builder
  @Builder
  private buildByDimension(): void {
    if (this.dimension === '2*4') {
      this.buildLarge()
    } else {
      this.buildMedium()
    }
  }

  // ===== 2×2 卡片布局 =====
  @Builder
  private buildMedium(): void {
    Column() {
      Text(this.title)
        .fontSize(10)
        .fontColor('#BBBBBB')

      Text(this.name)
        .fontSize(14)
        .fontColor('#FFFFFF')
        .margin({ top: 6 })

      Text(this.code)
        .fontSize(10)
        .fontColor('#666666')
        .margin({ top: 2 })

      Text(this.priceText)
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor(this.isUp() ? '#FF4D4F' : '#39D98A')
        .margin({ top: 8 })

      Text(`${this.changeText}  ${this.pctText}`)
        .fontSize(11)
        .fontColor(this.isUp() ? '#FF7875' : '#6BE49B')
        .margin({ top: 4 })

      Text(this.tsText)
        .fontSize(10)
        .fontColor('#666666')
        .margin({ top: 8 })
    }
    .padding(12)
    .width('100%')
    .height('100%')
  }

  // ===== 2×4 卡片布局 =====
  @Builder
  private buildLarge(): void {
    Column() {
      Row() {
        Column() {
          Text(this.title)
            .fontSize(10)
            .fontColor('#BBBBBB')

          Text(this.name)
            .fontSize(16)
            .fontColor('#FFFFFF')
            .margin({ top: 6 })

          Text(this.code)
            .fontSize(10)
            .fontColor('#666666')
            .margin({ top: 2 })
        }
        .layoutWeight(1)

        Column() {
          Text(this.priceText)
            .fontSize(28)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.isUp() ? '#FF4D4F' : '#39D98A')

          Text(`${this.changeText}  ${this.pctText}`)
            .fontSize(12)
            .fontColor(this.isUp() ? '#FF7875' : '#6BE49B')
            .margin({ top: 6 })

          Text(this.tsText)
            .fontSize(10)
            .fontColor('#666666')
            .margin({ top: 10 })
        }
      }
      .width('100%')
    }
    .padding(16)
    .width('100%')
    .height('100%')
  }
}
