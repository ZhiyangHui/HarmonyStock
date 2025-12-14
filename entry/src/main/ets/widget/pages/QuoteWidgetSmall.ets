const storage = new LocalStorage()

interface RouterParams {
  formId: string
}

interface RouterActionOption {
  action: 'router'
  abilityName: string
  params: RouterParams
}

declare function postCardAction(ctx: Object, option: RouterActionOption): void

type FormDimensionText = '1*2' | '2*2' | '2*4'

@Entry(storage)
@Component
struct QuoteWidget {
  @LocalStorageProp('formId') formId: string = ''
  @LocalStorageProp('title') title: string = ''
  @LocalStorageProp('code') code: string = ''
  @LocalStorageProp('name') name: string = ''
  @LocalStorageProp('priceText') priceText: string = '--'
  @LocalStorageProp('changeText') changeText: string = '--'
  @LocalStorageProp('pctText') pctText: string = '--'
  @LocalStorageProp('tsText') tsText: string = ''

  // 关键：由 FormExtensionAbility 在 updateForm 时写入
  @LocalStorageProp('dimension') dimension: FormDimensionText = '2*2'

  readonly abilityName: string = 'EntryAbility'

  private isUp(): boolean {
    if (this.changeText === '--') {
      return true
    }
    return !this.changeText.startsWith('-')
  }

  private openApp(): void {
    postCardAction(this, {
      action: 'router',
      abilityName: this.abilityName,
      params: { formId: this.formId }
    })
  }

  build() {
    Column() {
      // 用 Builder 分发，避免 switch 里直接“调用普通函数渲染”导致不符合 ArkUI 习惯
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

  @Builder
  private buildByDimension(): void {
    if (this.dimension === '1*2') {
      this.buildSmall()
    } else if (this.dimension === '2*4') {
      this.buildLarge()
    } else {
      this.buildMedium()
    }
  }

  // ===== 1×2 =====
  @Builder
  private buildSmall(): void {
    Column() {
      Row() {
        Text(this.name)
          .fontSize(13)
          .fontColor('#FFFFFF')
        Blank()
        Text(this.code)
          .fontSize(10)
          .fontColor('#666666')
      }
      .width('100%')

      Text(this.priceText)
        .fontSize(22)
        .fontWeight(FontWeight.Bold)
        .fontColor(this.isUp() ? '#FF4D4F' : '#39D98A')
        .margin({ top: 6 })

      Text(`${this.changeText}  ${this.pctText}`)
        .fontSize(10)
        .fontColor(this.isUp() ? '#FF7875' : '#6BE49B')
        .margin({ top: 4 })
    }
    .padding(12)
    .width('100%')
    .height('100%')
  }

  // ===== 2×2 =====
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

  // ===== 2×4 =====
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
